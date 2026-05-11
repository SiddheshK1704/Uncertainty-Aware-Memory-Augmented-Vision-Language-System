"""
UVLA Live Pipeline
==================

Controls:
---------
1 -> person
2 -> chair
3 -> bottle
4 -> laptop
5 -> phone

n -> Gaussian noise
b -> Motion blur
l -> Low light
o -> Occlusion
c -> Clear transforms

q -> Quit
"""

import time
import cv2
import numpy as np

from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from webcam.hud_renderer import HUDRenderer
from webcam.command_input import CommandInputHandler

# ============================================================
# CONFIG
# ============================================================

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

YOLO_CONFIDENCE = 0.4
UNCERTAINTY_THRESHOLD = 0.40

# ============================================================
# LOAD MODELS
# ============================================================

print("[INFO] Loading YOLOv8...")
yolo_model = YOLO("yolov8n.pt")

print("[INFO] Loading language model...")
text_model = SentenceTransformer("all-MiniLM-L6-v2")

# ============================================================
# TRANSFORMS
# ============================================================

def add_gaussian_noise(frame, std=25):
    noise = np.random.normal(0, std, frame.shape).astype(np.float32)
    return np.clip(
        frame.astype(np.float32) + noise,
        0,
        255
    ).astype(np.uint8)


def add_motion_blur(frame, ksize=15):

    kernel = np.zeros((ksize, ksize), dtype=np.float32)

    kernel[ksize // 2, :] = 1.0 / ksize

    return cv2.filter2D(frame, -1, kernel)


def add_low_light(frame, gamma=2.5):

    table = np.array([
        ((i / 255.0) ** (1.0 / gamma)) * 255
        for i in range(256)
    ]).astype(np.uint8)

    return cv2.LUT(frame, table)


def add_occlusion(frame, num=3, size=100):

    out = frame.copy()

    h, w = frame.shape[:2]

    for _ in range(num):

        x = np.random.randint(0, w - size)
        y = np.random.randint(0, h - size)

        out[y:y+size, x:x+size] = 0

    return out


# ============================================================
# MEMORY
# ============================================================

memory = {}

# ============================================================
# PERCEPTION
# ============================================================

def enhance_frame(frame):

    alpha = 1.15
    beta = 12

    enhanced = cv2.convertScaleAbs(
        frame,
        alpha=alpha,
        beta=beta
    )

    return enhanced


# ============================================================
# UNCERTAINTY
# ============================================================

def compute_uncertainty(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    lap_var = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    brightness = np.mean(gray)

    lap_score = min(lap_var / 1000.0, 1.0)
    brightness_score = min(brightness / 128.0, 1.0)

    confidence = (
        0.6 * lap_score +
        0.4 * brightness_score
    )

    confidence = float(
        np.clip(confidence, 0.0, 1.0)
    )

    return confidence, lap_var


# ============================================================
# LANGUAGE GROUNDING
# ============================================================

def ground_command(command, detections):

    if not command:
        return None

    if len(detections) == 0:
        return None

    labels = [d["label"] for d in detections]

    command_embedding = text_model.encode([command])

    label_embeddings = text_model.encode(labels)

    similarities = cosine_similarity(
        command_embedding,
        label_embeddings
    )[0]

    best_idx = int(np.argmax(similarities))

    best_score = similarities[best_idx]

    if best_score < 0.25:
        return None

    return detections[best_idx]["label"]


# ============================================================
# DECISION
# ============================================================

def decide_action(found, confidence, offset_x):

    if confidence < UNCERTAINTY_THRESHOLD:
        return "LOW_CONF"

    if not found:
        return "SEARCHING"

    if offset_x < -100:
        return "TURN_LEFT"

    elif offset_x > 100:
        return "TURN_RIGHT"

    return "MOVE_FORWARD"


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# ============================================================
# HUD
# ============================================================

hud = HUDRenderer(
    frame_w=FRAME_WIDTH,
    frame_h=FRAME_HEIGHT
)

# ============================================================
# COMMAND HANDLER
# ============================================================

cmd_handler = CommandInputHandler(
    default_command="find chair"
)

cmd_handler.start()

# ============================================================
# RUNTIME STATE
# ============================================================

prev_time = time.time()

transform_mode = "none"

quick_targets = {
    ord("1"): "find person",
    ord("2"): "find chair",
    ord("3"): "find bottle",
    ord("4"): "find laptop",
    ord("5"): "find phone",
}

# ============================================================
# MAIN LOOP
# ============================================================

print("[INFO] Starting UVLA live system...")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # --------------------------------------------------------
    # QUICK TARGETS
    # --------------------------------------------------------

    command = cmd_handler.get_command()

    # --------------------------------------------------------
    # TRANSFORMS
    # --------------------------------------------------------

    if transform_mode == "noise":
        frame = add_gaussian_noise(frame)

    elif transform_mode == "blur":
        frame = add_motion_blur(frame)

    elif transform_mode == "low_light":
        frame = add_low_light(frame)

    elif transform_mode == "occlusion":
        frame = add_occlusion(frame)

    # --------------------------------------------------------
    # PERCEPTION
    # --------------------------------------------------------

    enhanced_frame = enhance_frame(frame)

    # --------------------------------------------------------
    # UNCERTAINTY
    # --------------------------------------------------------

    unc_confidence, lap_var = compute_uncertainty(
        enhanced_frame
    )

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    results = yolo_model(
        enhanced_frame,
        conf=YOLO_CONFIDENCE,
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

            label = yolo_model.names[cls_id]

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "center": (cx, cy),
            })

            memory[label] = {
                "bbox": [x1, y1, x2, y2],
                "timestamp": time.time()
            }

    # --------------------------------------------------------
    # GROUNDING
    # --------------------------------------------------------

    target_label = ground_command(
        command,
        detections
    )

    target_detection = None

    if target_label:

        for d in detections:

            if d["label"] == target_label:
                target_detection = d
                break

    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    from_memory = False
    memory_entries = []

    if (
        target_detection is None and
        target_label in memory
    ):

        from_memory = True

        mem = memory[target_label]

        memory_entries.append(
            type(
                "MemoryEntry",
                (),
                {
                    "label": target_label,
                    "bbox": mem["bbox"],
                    "age_seconds": (
                        time.time() -
                        mem["timestamp"]
                    )
                }
            )
        )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    found = target_detection is not None

    offset_x = 0

    if found:

        offset_x = (
            target_detection["center"][0] -
            FRAME_WIDTH // 2
        )

    action_status = decide_action(
        found,
        unc_confidence,
        offset_x
    )

    # --------------------------------------------------------
    # HUD DETECTIONS
    # --------------------------------------------------------

    hud_detections = []

    for d in detections:

        hud_detections.append(
            type(
                "Detection",
                (),
                {
                    "label": d["label"],
                    "confidence": d["confidence"],
                    "bbox": d["bbox"],
                    "center": d["center"]
                }
            )
        )

    # --------------------------------------------------------
    # HUD RENDER
    # --------------------------------------------------------

    final_frame = hud.draw_frame(
        frame=enhanced_frame,
        detections=hud_detections,
        target_label=target_label,
        command=command,
        unc_confidence=unc_confidence,
        unc_lap_var=lap_var,
        action_status=action_status,
        action_label=target_label,
        memory_entries=memory_entries,
        from_memory=from_memory,
        perception_tags=[transform_mode],
    )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    now = time.time()

    fps = 1.0 / max(now - prev_time, 1e-6)

    prev_time = now

    cv2.putText(
        final_frame,
        f"FPS: {fps:.1f}",
        (20, FRAME_HEIGHT - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # HELP TEXT
    # --------------------------------------------------------

    cv2.putText(
        final_frame,
        "N:Noise B:Blur L:LowLight O:Occlusion C:Clear",
        (20, FRAME_HEIGHT - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    # --------------------------------------------------------
    # SHOW
    # --------------------------------------------------------

    cv2.imshow(
        "UVLA Live System",
        final_frame
    )

    # --------------------------------------------------------
    # KEYBOARD CONTROLS
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("n"):
        transform_mode = "noise"

    elif key == ord("b"):
        transform_mode = "blur"

    elif key == ord("l"):
        transform_mode = "low_light"

    elif key == ord("o"):
        transform_mode = "occlusion"

    elif key == ord("c"):
        transform_mode = "none"

    elif key in quick_targets:

        new_command = quick_targets[key]

        cmd_handler._current_command = new_command

        print(f"[TARGET] {new_command}")

# ============================================================
# CLEANUP
# ============================================================

cmd_handler.stop()

cap.release()

cv2.destroyAllWindows()