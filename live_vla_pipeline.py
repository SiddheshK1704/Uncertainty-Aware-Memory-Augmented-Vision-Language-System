import cv2
import time
import math
import json
import queue
import random
import threading
import numpy as np

from ultralytics import YOLO
from groq import Groq
from dotenv import load_dotenv
import os

# =========================================================
# LOAD ENV
# =========================================================

load_dotenv()

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# =========================================================
# CONFIG
# =========================================================

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

YOLO_CONF = 0.45

CONFIDENCE_THRESHOLD = 0.45

CENTER_TOLERANCE = 120

MEMORY_TIMEOUT = 120

# =========================================================
# LOAD MODEL
# =========================================================

model = YOLO("yolov8n.pt")

# =========================================================
# GLOBAL STATE
# =========================================================

instruction_text = "find the chair"

parsed_instruction = {
    "target_object": "chair",
    "action": "track",
    "spatial_relation": None
}

memory_bank = {}

current_transform = "none"

# =========================================================
# GROQ LLM PARSER
# =========================================================

def parse_instruction_with_groq(text):

    prompt = f"""
    Convert this robotics instruction into JSON.

    Instruction:
    "{text}"

    Return ONLY valid JSON.

    Format:
    {{
        "target_object": "",
        "action": "",
        "spatial_relation": ""
    }}
    """

    try:

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        content = response.choices[0].message.content

        parsed = json.loads(content)

        return parsed

    except Exception as e:
        print("LLM Error:", e)

        return {
            "target_object": "chair",
            "action": "track",
            "spatial_relation": None
        }

# =========================================================
# TRANSFORMS
# =========================================================

def add_noise(frame):

    noise = np.random.normal(
        0,
        25,
        frame.shape
    ).astype(np.float32)

    return np.clip(
        frame.astype(np.float32) + noise,
        0,
        255
    ).astype(np.uint8)


def add_blur(frame):

    return cv2.GaussianBlur(
        frame,
        (11, 11),
        0
    )


def add_low_light(frame):

    return np.clip(
        frame * 0.35,
        0,
        255
    ).astype(np.uint8)


def add_occlusion(frame):

    h, w = frame.shape[:2]

    x = random.randint(0, w - 250)
    y = random.randint(0, h - 250)

    frame[y:y+250, x:x+250] = 0

    return frame


# =========================================================
# ADAPTIVE PERCEPTION
# =========================================================

def adaptive_enhancement(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)

    enhanced = frame.copy()

    if brightness < 60:

        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(8, 8)
        )

        cl = clahe.apply(l)

        limg = cv2.merge((cl, a, b))

        enhanced = cv2.cvtColor(
            limg,
            cv2.COLOR_LAB2BGR
        )

    return enhanced

# =========================================================
# UNCERTAINTY MODULE
# =========================================================

def compute_uncertainty(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    lap_var = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    brightness = np.mean(gray)

    sharpness_score = min(
        lap_var / 1000.0,
        1.0
    )

    brightness_score = min(
        brightness / 120.0,
        1.0
    )

    confidence = (
        0.7 * sharpness_score +
        0.3 * brightness_score
    )

    confidence = max(
        0.0,
        min(confidence, 1.0)
    )

    return confidence, lap_var

# =========================================================
# MEMORY MODULE
# =========================================================

def update_memory(detections, step):

    global memory_bank

    for d in detections:

        memory_bank[d["name"]] = {
            "cx": d["cx"],
            "cy": d["cy"],
            "bbox": d["bbox"],
            "timestamp": step,
            "confidence": d["confidence"]
        }

    expired = []

    for k, v in memory_bank.items():

        if step - v["timestamp"] > MEMORY_TIMEOUT:
            expired.append(k)

    for e in expired:
        del memory_bank[e]

# =========================================================
# ACTION LOGIC
# =========================================================

def decide_action(found, offset_x, confidence):

    if confidence < CONFIDENCE_THRESHOLD:
        return "RE-EVALUATE"

    if not found:
        return "SCANNING"

    if offset_x < -CENTER_TOLERANCE:
        return "TURN LEFT"

    elif offset_x > CENTER_TOLERANCE:
        return "TURN RIGHT"

    return "MOVE FORWARD"

# =========================================================
# DRAWING
# =========================================================

def draw_hud(
    frame,
    fps,
    confidence,
    action,
    target,
    transform
):

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (420, 230),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.5,
        frame,
        0.5,
        0,
        frame
    )

    green = (0, 255, 120)

    cv2.putText(
        frame,
        "ROBUST VLA PIPELINE",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        green,
        2
    )

    lines = [
        f"FPS: {fps:.1f}",
        f"Target: {target}",
        f"Action: {action}",
        f"Confidence: {confidence:.2f}",
        f"Transform: {transform}",
        f"Memory Objects: {len(memory_bank)}"
    ]

    y = 70

    for line in lines:

        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        y += 30

# =========================================================
# MAIN LOOP
# =========================================================

def main():

    global instruction_text
    global parsed_instruction
    global current_transform

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Camera not available")
        return

    print("=" * 60)
    print("ROBUST VLA PIPELINE")
    print("=" * 60)

    print("\nControls:")
    print("q -> Quit")
    print("i -> Enter instruction")
    print("t -> Cycle transforms")
    print("1 -> none")
    print("2 -> noise")
    print("3 -> blur")
    print("4 -> low light")
    print("5 -> occlusion")

    transforms = [
        "none",
        "noise",
        "blur",
        "low_light",
        "occlusion"
    ]

    step = 0

    prev_time = time.time()

    fps = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ============================================
        # PERTURBATIONS
        # ============================================

        if current_transform == "noise":
            frame = add_noise(frame)

        elif current_transform == "blur":
            frame = add_blur(frame)

        elif current_transform == "low_light":
            frame = add_low_light(frame)

        elif current_transform == "occlusion":
            frame = add_occlusion(frame)

        # ============================================
        # ADAPTIVE PERCEPTION
        # ============================================

        frame = adaptive_enhancement(frame)

        # ============================================
        # UNCERTAINTY
        # ============================================

        confidence, lap_var = compute_uncertainty(frame)

        # ============================================
        # DETECTION
        # ============================================

        results = model(
            frame,
            conf=YOLO_CONF,
            verbose=False
        )

        detections = []

        for r in results:

            for i in range(len(r.boxes)):

                cls_id = int(r.boxes.cls[i].item())

                conf = float(r.boxes.conf[i].item())

                x1, y1, x2, y2 = (
                    r.boxes.xyxy[i]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                name = model.names[cls_id]

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                detections.append({
                    "name": name.lower(),
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "cx": cx,
                    "cy": cy
                })

        # ============================================
        # MEMORY
        # ============================================

        update_memory(
            detections,
            step
        )

        # ============================================
        # GROUNDING
        # ============================================

        target = parsed_instruction["target_object"]

        best_target = None

        best_conf = 0

        for d in detections:

            if target in d["name"]:

                if d["confidence"] > best_conf:

                    best_conf = d["confidence"]

                    best_target = d

        found = best_target is not None

        offset_x = 0

        # ============================================
        # DRAW OBJECTS
        # ============================================

        for d in detections:

            x1, y1, x2, y2 = d["bbox"]

            is_target = (
                best_target is not None and
                d == best_target
            )

            color = (
                (0, 255, 120)
                if is_target
                else
                (160, 160, 160)
            )

            thickness = 3 if is_target else 1

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                thickness
            )

            label = (
                f"{d['name']} "
                f"{d['confidence']:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            if is_target:

                offset_x = (
                    d["cx"] -
                    frame.shape[1] // 2
                )

                cv2.circle(
                    frame,
                    (d["cx"], d["cy"]),
                    10,
                    (0, 255, 255),
                    -1
                )

                cv2.putText(
                    frame,
                    "TARGET",
                    (x1, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

        # ============================================
        # MEMORY RECALL
        # ============================================

        if not found and target in memory_bank:

            mem = memory_bank[target]

            cv2.putText(
                frame,
                "MEMORY RECALL ACTIVE",
                (50, FRAME_HEIGHT - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 180, 255),
                3
            )

            cv2.circle(
                frame,
                (mem["cx"], mem["cy"]),
                40,
                (0, 180, 255),
                3
            )

        # ============================================
        # ACTION
        # ============================================

        action = decide_action(
            found,
            offset_x,
            confidence
        )

        # ============================================
        # GUIDE ARROWS
        # ============================================

        center_x = FRAME_WIDTH // 2
        center_y = FRAME_HEIGHT // 2

        if action == "TURN LEFT":

            cv2.arrowedLine(
                frame,
                (center_x, center_y),
                (center_x - 150, center_y),
                (0, 255, 255),
                6
            )

        elif action == "TURN RIGHT":

            cv2.arrowedLine(
                frame,
                (center_x, center_y),
                (center_x + 150, center_y),
                (0, 255, 255),
                6
            )

        elif action == "MOVE FORWARD":

            cv2.arrowedLine(
                frame,
                (center_x, center_y + 120),
                (center_x, center_y - 120),
                (0, 255, 120),
                6
            )

        elif action == "RE-EVALUATE":

            cv2.putText(
                frame,
                "LOW CONFIDENCE",
                (450, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                4
            )

        # ============================================
        # FPS
        # ============================================

        now = time.time()

        dt = now - prev_time

        fps = 0.9 * fps + 0.1 * (1 / dt)

        prev_time = now

        # ============================================
        # HUD
        # ============================================

        draw_hud(
            frame,
            fps,
            confidence,
            action,
            target,
            current_transform
        )

        # ============================================
        # SHOW
        # ============================================

        cv2.imshow(
            "Robust VLA Pipeline",
            frame
        )

        # ============================================
        # KEYBOARD
        # ============================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("i"):

            text = input("\nInstruction: ")

            if text.strip():

                instruction_text = text

                parsed_instruction = (
                    parse_instruction_with_groq(text)
                )

                print("\nParsed:")
                print(parsed_instruction)

        elif key == ord("1"):
            current_transform = "none"

        elif key == ord("2"):
            current_transform = "noise"

        elif key == ord("3"):
            current_transform = "blur"

        elif key == ord("4"):
            current_transform = "low_light"

        elif key == ord("5"):
            current_transform = "occlusion"

        step += 1

    cap.release()

    cv2.destroyAllWindows()

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()