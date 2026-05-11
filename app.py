"""
UVLA Streamlit Application
===========================
Production-grade UI for the Uncertainty-Aware Memory-Augmented
Vision-Language-Action System.

Run with:
    streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import json
import time
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional

# ---- Page config (MUST be first Streamlit call) ----
st.set_page_config(
    page_title="UVLA System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS ----
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
  }

  .stApp {
    background: #0d1117;
  }

  /* Header */
  .main-header {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    border: 1px solid #1e3a4f;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  }
  .main-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    color: #e0f4ff;
    margin: 0;
    letter-spacing: -0.5px;
  }
  .main-header p {
    color: #7ecbf7;
    margin: 6px 0 0;
    font-size: 0.95rem;
  }

  /* Module cards */
  .module-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 18px 20px;
    margin: 8px 0;
    transition: border-color 0.2s;
  }
  .module-card:hover { border-color: #388bfd; }
  .module-card h4 {
    font-family: 'JetBrains Mono', monospace;
    color: #58a6ff;
    font-size: 0.85rem;
    margin: 0 0 8px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .module-card p { color: #8b949e; font-size: 0.88rem; margin: 0; line-height: 1.5; }

  /* Status badges */
  .badge-success { background: #1a4a2e; color: #56d364; padding: 3px 10px; border-radius: 20px; font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; }
  .badge-warning { background: #3d2b00; color: #e3b341; padding: 3px 10px; border-radius: 20px; font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; }
  .badge-danger  { background: #3d1a1a; color: #f85149; padding: 3px 10px; border-radius: 20px; font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; }
  .badge-info    { background: #1a2d4a; color: #79c0ff; padding: 3px 10px; border-radius: 20px; font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; }

  /* Metric value */
  .metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #e6edf3;
    line-height: 1;
  }
  .metric-label {
    font-size: 0.78rem;
    color: #6e7681;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 4px;
  }

  /* Detection box */
  .detection-item {
    background: #161b22;
    border-left: 3px solid #388bfd;
    padding: 10px 14px;
    border-radius: 0 8px 8px 0;
    margin: 4px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #c9d1d9;
  }
  .detection-target {
    border-left-color: #3fb950;
    background: #0d2818;
  }

  /* Action result */
  .action-box-success {
    background: linear-gradient(135deg, #0d2818, #162032);
    border: 1px solid #3fb950;
    border-radius: 12px;
    padding: 20px 24px;
  }
  .action-box-reject {
    background: linear-gradient(135deg, #2d1315, #1a1a1a);
    border: 1px solid #f85149;
    border-radius: 12px;
    padding: 20px 24px;
  }
  .action-box-memory {
    background: linear-gradient(135deg, #2d2200, #1a1a1a);
    border: 1px solid #e3b341;
    border-radius: 12px;
    padding: 20px 24px;
  }
  .action-type {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 1px;
  }

  /* Memory panel */
  .memory-entry {
    background: #1c2128;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 4px 0;
    font-size: 0.82rem;
    color: #8b949e;
    border: 1px solid #2d333b;
    font-family: 'JetBrains Mono', monospace;
  }

  /* Sidebar */
  .css-1d391kg { background: #0d1117; }

  /* Remove streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }

  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    background: #161b22;
    border-radius: 8px 8px 0 0;
    color: #8b949e;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
  }
  .stTabs [aria-selected="true"] {
    background: #1f2937;
    color: #58a6ff;
  }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Pipeline initialization (cached for performance)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load all UVLA modules once and cache them."""
    from modules import (
        VisionModule, LanguageModule, GroundingModule,
        UncertaintyModule, MemoryModule, PerceptionModule, DecisionModule,
    )
    with st.spinner("🔧 Loading UVLA pipeline modules..."):
        vision      = VisionModule(model_size="n", conf_threshold=0.30)
        language    = LanguageModule()
        grounding   = GroundingModule(language_module=language)
        uncertainty = UncertaintyModule(blur_threshold=60.0)
        memory      = MemoryModule(ttl=120.0)
        perception  = PerceptionModule(auto_enhance=True)
        decision    = DecisionModule(
            memory_module=memory,
            confidence_threshold=0.25,
        )
    return vision, language, grounding, uncertainty, memory, perception, decision


def pil_to_bgr(pil_img) -> np.ndarray:
    """Convert PIL image to BGR numpy array."""
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert BGR numpy array to PIL Image."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def apply_perturbation(image: np.ndarray, condition: str) -> np.ndarray:
    """Apply a named perturbation to the image."""
    from utils.image_utils import (
        apply_gaussian_noise, apply_blur, apply_low_light, apply_occlusion
    )
    if condition == "Gaussian Noise":
        return apply_gaussian_noise(image, sigma=25.0)
    elif condition == "Blur":
        return apply_blur(image, kernel_size=15)
    elif condition == "Low Light":
        return apply_low_light(image, gamma=0.3)
    elif condition == "Occlusion":
        return apply_occlusion(image, occlusion_fraction=0.30)
    return image


def make_confidence_gauge(value: float, title: str, color: str = "#58a6ff") -> go.Figure:
    """Create a Plotly gauge chart for a confidence metric."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%", "font": {"size": 28, "color": "#e6edf3", "family": "JetBrains Mono"}},
        title={"text": title, "font": {"size": 13, "color": "#8b949e", "family": "Syne"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#444", "tickfont": {"color": "#555"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#161b22",
            "bordercolor": "#21262d",
            "steps": [
                {"range": [0, 30],  "color": "#2d1315"},
                {"range": [30, 60], "color": "#2d2200"},
                {"range": [60, 100],"color": "#0d2818"},
            ],
            "threshold": {
                "line": {"color": "#f0f0f0", "width": 2},
                "thickness": 0.8,
                "value": 50,
            },
        },
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e6edf3",
    )
    return fig


def make_grounding_bar_chart(all_scores: list) -> go.Figure:
    """Create a horizontal bar chart for grounding similarity scores."""
    if not all_scores:
        return go.Figure()

    labels = [s["label"] for s in all_scores[:6]]
    scores = [s["score"] for s in all_scores[:6]]
    colors = ["#3fb950" if i == 0 else "#388bfd" for i in range(len(labels))]

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{s:.3f}" for s in scores],
        textposition="outside",
        textfont={"color": "#e6edf3", "size": 11, "family": "JetBrains Mono"},
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=60, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 1.1], gridcolor="#21262d", color="#8b949e"),
        yaxis=dict(gridcolor="#21262d", color="#e6edf3"),
        title=dict(text="Grounding Similarity Scores", font=dict(color="#8b949e", size=12)),
    )
    return fig


def run_full_pipeline(
    image_bgr: np.ndarray,
    command: str,
    perturbation: str,
    vision, language, grounding, uncertainty, memory, perception, decision
) -> dict:
    """Execute the full UVLA pipeline and return all intermediate results."""
    from utils.visualization import VisualizationUtils

    # Apply perturbation
    perturbed = apply_perturbation(image_bgr, perturbation)

    # 1. Adaptive Perception
    perc_result = perception.enhance(perturbed)
    enhanced = perc_result.enhanced_image

    # 2. Uncertainty Estimation
    unc_result = uncertainty.estimate(enhanced)

    # 3. Object Detection
    detections = vision.detect(enhanced)

    # 4. Memory Update
    memory.update(detections)

    # 5. Language Grounding
    ground_result = grounding.ground(command, detections)

    # 6. Decision
    action = decision.execute(command, ground_result, unc_result)

    # 7. Build annotated image
    viz = VisualizationUtils()
    annotated = viz.draw_detections(enhanced, detections, ground_result)
    annotated = viz.draw_action_overlay(annotated, action)
    annotated = viz.draw_confidence_hud(
        annotated,
        quality_score=unc_result.overall_confidence,
        grounding_score=ground_result.similarity_score,
        combined_score=action.confidence,
    )

    # Memory overlay
    mem_entries = memory.recall_all()
    if mem_entries:
        annotated = viz.draw_memory_overlay(annotated, mem_entries, alpha=0.15)

    return {
        "original":        image_bgr,
        "perturbed":       perturbed,
        "enhanced":        enhanced,
        "annotated":       annotated,
        "uncertainty":     unc_result,
        "detections":      detections,
        "grounding":       ground_result,
        "action":          action,
        "perception":      perc_result,
        "memory_entries":  mem_entries,
    }


# ============================================================
# Main App Layout
# ============================================================

def main():
    # ---- Header ----
    st.markdown("""
    <div class="main-header">
        <h1>🤖 UVLA System</h1>
        <p>Uncertainty-Aware &amp; Memory-Augmented Vision-Language-Action Pipeline</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Load pipeline ----
    try:
        vision, language, grounding, uncertainty, memory, perception, decision = load_pipeline()
        pipeline_ok = True
    except Exception as e:
        st.error(f"❌ Failed to load pipeline: {e}")
        st.info("Make sure all dependencies are installed: `pip install -r requirements.txt`")
        pipeline_ok = False
        return

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("### ⚙️ Pipeline Settings")

        perturbation = st.selectbox(
            "Simulated Condition",
            ["Clean", "Gaussian Noise", "Blur", "Low Light", "Occlusion"],
            help="Apply a distribution shift to test robustness",
        )

        st.markdown("---")
        st.markdown("### 🔧 Thresholds")

        conf_thresh = st.slider(
            "Confidence Gate", 0.0, 1.0, 0.25, 0.05,
            help="Minimum confidence to allow action execution",
        )
        decision.confidence_threshold = conf_thresh

        blur_thresh = st.slider(
            "Blur Rejection Threshold", 10.0, 300.0, 60.0, 10.0,
            help="Laplacian variance below this rejects the image",
        )
        uncertainty.blur_threshold = blur_thresh

        st.markdown("---")
        st.markdown("### 🧠 Memory")
        if st.button("🗑️ Clear Memory", use_container_width=True):
            memory.clear()
            st.success("Memory cleared!")

        mem_summary = memory.summary()
        st.caption(f"Stored objects: **{mem_summary['total_entries']}**")
        for label in mem_summary["labels"]:
            st.markdown(f'<div class="memory-entry">📌 {label}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📖 Architecture")
        st.markdown("""
        ```
        Image → Perception → Uncertainty
                          ↓
                    Vision (YOLO)
                          ↓
        Command → Language → Grounding
                          ↓
                    Memory ← ↕
                          ↓
                    Decision Gate
                          ↓
                       Action
        ```
        """)

    # ---- Main content area ----
    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("### 📤 Input")

        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Upload any image containing objects",
        )

        command = st.text_input(
            "💬 Natural Language Command",
            value="navigate to the chair",
            placeholder="e.g. 'pick up the bottle', 'find the tv'",
        )

        # Example commands
        st.caption("Quick examples:")
        example_cols = st.columns(3)
        example_commands = [
            "navigate to the chair",
            "find the tv",
            "pick up the bottle",
        ]
        for i, ex in enumerate(example_commands):
            if example_cols[i].button(ex, use_container_width=True, key=f"ex_{i}"):
                command = ex
                st.rerun()

        # Show original or demo image
        if uploaded_file:
            pil_img = Image.open(uploaded_file)
            st.image(pil_img, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
            image_bgr = pil_to_bgr(pil_img)
        else:
            # Generate a demo image
            from scripts.demo import create_demo_image
            image_bgr = create_demo_image()
            st.image(
                bgr_to_pil(image_bgr),
                caption="Demo Image (upload your own above)",
                use_container_width=True,
            )

        # Show condition preview
        if perturbation != "Clean":
            st.markdown(f"**Preview — {perturbation}:**")
            perturbed_preview = apply_perturbation(image_bgr, perturbation)
            st.image(bgr_to_pil(perturbed_preview), use_container_width=True)

        run_btn = st.button(
            "🚀 Run Pipeline",
            use_container_width=True,
            type="primary",
        )

    with col_output:
        st.markdown("### 📊 Results")

        if run_btn and command.strip():
            with st.spinner("Running UVLA pipeline..."):
                t0 = time.time()
                results = run_full_pipeline(
                    image_bgr, command, perturbation,
                    vision, language, grounding, uncertainty, memory, perception, decision,
                )
                elapsed = time.time() - t0

            # ---- Annotated image ----
            st.image(
                bgr_to_pil(results["annotated"]),
                caption=f"Annotated Output  ({elapsed*1000:.0f}ms)",
                use_container_width=True,
            )

            # ---- Action Result ----
            action = results["action"]
            action_type = action.action_type.value

            if action.succeeded and not action.from_memory:
                box_class = "action-box-success"
                icon = "✅"
                badge = '<span class="badge-success">EXECUTED</span>'
            elif action.from_memory:
                box_class = "action-box-memory"
                icon = "🧠"
                badge = '<span class="badge-warning">MEMORY FALLBACK</span>'
            else:
                box_class = "action-box-reject"
                icon = "🛑"
                badge = '<span class="badge-danger">REJECTED</span>'

            st.markdown(f"""
            <div class="{box_class}">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <span class="action-type">{icon} {action_type.upper()}</span>
                    {badge}
                </div>
                <div style="color:#c9d1d9; font-size:0.9rem; line-height:1.6;">
                    <b>Target:</b> {action.target_label or "None"}<br>
                    <b>Confidence:</b> {action.confidence:.3f}<br>
                    <b>From Memory:</b> {"Yes" if action.from_memory else "No"}<br>
                    <b>Reason:</b> {action.reason}
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif run_btn and not command.strip():
            st.warning("Please enter a command first.")
        else:
            st.info("Configure inputs and click **Run Pipeline** to see results.")

    # ---- Detail tabs (shown after run) ----
    if run_btn and command.strip() and "results" in dir():
        st.markdown("---")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Detections",
            "📊 Confidence",
            "🎯 Grounding",
            "🧠 Memory",
            "🖼️ Image Stages",
        ])

        with tab1:
            st.markdown("#### Detected Objects")
            dets = results["detections"]
            if dets:
                target_label = (
                    results["grounding"].target_detection.label
                    if results["grounding"].grounded and results["grounding"].target_detection
                    else None
                )
                for det in dets:
                    is_target = det.label == target_label
                    card_class = "detection-item detection-target" if is_target else "detection-item"
                    star = " ⭐ TARGET" if is_target else ""
                    st.markdown(
                        f'<div class="{card_class}">'
                        f'<b>{det.label}{star}</b>  '
                        f'conf={det.confidence:.3f}  '
                        f'bbox=[{", ".join(str(round(v)) for v in det.bbox)}]'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("No objects detected. Try adjusting the confidence threshold.")

            # Uncertainty details
            st.markdown("#### Image Quality Signals")
            unc = results["uncertainty"]
            ucols = st.columns(4)
            ucols[0].metric("Laplacian Var", f"{unc.laplacian_variance:.1f}")
            ucols[1].metric("Sharpness",     f"{unc.normalized_sharpness:.3f}")
            ucols[2].metric("Brightness",    f"{unc.brightness_score:.3f}")
            ucols[3].metric("Overall",       f"{unc.overall_confidence:.3f}")

            if not unc.is_acceptable:
                st.error(f"⚠️ Image rejected: {unc.rejection_reason}")

        with tab2:
            g1, g2, g3 = st.columns(3)
            action = results["action"]
            unc = results["uncertainty"]
            ground = results["grounding"]

            with g1:
                st.plotly_chart(
                    make_confidence_gauge(unc.overall_confidence, "Image Quality", "#58a6ff"),
                    use_container_width=True, key="gauge_quality"
                )
            with g2:
                st.plotly_chart(
                    make_confidence_gauge(ground.similarity_score, "Grounding Score", "#3fb950"),
                    use_container_width=True, key="gauge_grounding"
                )
            with g3:
                st.plotly_chart(
                    make_confidence_gauge(action.confidence, "Combined", "#e3b341"),
                    use_container_width=True, key="gauge_combined"
                )

            # Perception transforms
            perc = results["perception"]
            st.markdown("#### Adaptive Perception")
            if perc.applied_transforms:
                for t in perc.applied_transforms:
                    st.markdown(f'<span class="badge-info">{t}</span> ', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-success">No enhancement needed</span>', unsafe_allow_html=True)

        with tab3:
            ground = results["grounding"]
            st.markdown(f"**Command:** `{ground.command}`")
            st.markdown(f"**Grounded:** {'✅ Yes' if ground.grounded else '❌ No'}")
            if ground.all_scores:
                fig = make_grounding_bar_chart(
                    [{"label": l, "score": s} for l, s in ground.all_scores]
                )
                st.plotly_chart(fig, use_container_width=True, key="grounding_bars")
            else:
                st.warning("No grounding scores available (no detections).")

        with tab4:
            mem_entries = results["memory_entries"]
            if mem_entries:
                st.markdown(f"**{len(mem_entries)} object(s) in memory:**")
                for entry in mem_entries:
                    st.markdown(
                        f'<div class="memory-entry">'
                        f'📍 <b>{entry.label}</b> '
                        f'— center=({entry.center[0]:.0f}, {entry.center[1]:.0f}) '
                        f'— {entry.age_seconds:.1f}s ago '
                        f'— seen {entry.seen_count}×'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Memory is empty. Run the pipeline a few times to build up memory.")

            # JSON dump
            with st.expander("Raw Memory JSON"):
                st.json(memory.summary())

        with tab5:
            img_cols = st.columns(3)
            img_cols[0].image(bgr_to_pil(results["original"]),   caption="Original",  use_container_width=True)
            img_cols[1].image(bgr_to_pil(results["perturbed"]),  caption="Perturbed", use_container_width=True)
            img_cols[2].image(bgr_to_pil(results["enhanced"]),   caption="Enhanced",  use_container_width=True)

            # Full JSON results
            with st.expander("📄 Full Pipeline JSON Output"):
                output_dict = {
                    "command": command,
                    "perturbation": perturbation,
                    "uncertainty": results["uncertainty"].to_dict(),
                    "detections": [d.to_dict() for d in results["detections"]],
                    "grounding": results["grounding"].to_dict(),
                    "action": results["action"].to_dict(),
                    "perception": results["perception"].to_dict(),
                }
                st.json(output_dict)

                # Download button
                st.download_button(
                    "⬇️ Download JSON",
                    data=json.dumps(output_dict, indent=2),
                    file_name="uvla_output.json",
                    mime="application/json",
                )

    # ---- Footer ----
    st.markdown("---")
    st.markdown(
        '<p style="color:#484f58; font-size:0.78rem; text-align:center; font-family:JetBrains Mono">'
        'UVLA System · YOLOv8 + MiniLM + Laplacian Uncertainty + Memory Module'
        '</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()