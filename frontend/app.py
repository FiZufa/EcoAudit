from pathlib import Path
import io
import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent.parent / "training/runs/detect/runs/ecoaudit_v17/weights/best.pt"

CLASS_COLORS = {
    "Battery":       "#FF6347",
    "Biological":    "#3CB371",
    "Cable":         "#4682B4",
    "Can":           "#FFA500",
    "Cardboard":     "#8B5A2B",
    "Clothes":       "#DA70D6",
    "E-waste":       "#FFD700",
    "Glass":         "#6495ED",
    "Medical Waste": "#DC143C",
    "Metal":         "#A9A9A9",
    "Paper":         "#90EE90",
    "Plastic":       "#1E90FF",
    "Shoes":         "#FFB6C1",
    "Trash":         "#800080",
}

st.set_page_config(
    page_title="EcoAudit — Waste Detection",
    page_icon="♻️",
    layout="wide",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2E7D32;
    }
    .subtitle {
        color: #666;
        margin-bottom: 1.5rem;
    }
    .detection-card {
        background: #f8f9fa;
        border-left: 4px solid;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .conf-bar-bg {
        background: #e0e0e0;
        border-radius: 999px;
        height: 8px;
        width: 100%;
    }
    .section-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #888;
        margin-bottom: 0.5rem;
    }
    [data-testid="stFileUploader"] {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


def run_detection(model, image: Image.Image, conf: float):
    results = model.predict(image, conf=conf, verbose=False)
    return results[0]


def get_annotated_image(result) -> Image.Image:
    annotated = result.plot()  # BGR numpy array
    return Image.fromarray(annotated[:, :, ::-1])  # to RGB


def build_detections(result) -> list[dict]:
    detections = []
    for box in result.boxes:
        cls_name = result.names[int(box.cls)]
        detections.append({
            "class_name": cls_name,
            "confidence": round(float(box.conf), 4),
            "color": CLASS_COLORS.get(cls_name, "#607D8B"),
            "bbox": {
                "x1": round(float(box.xyxy[0][0]), 1),
                "y1": round(float(box.xyxy[0][1]), 1),
                "x2": round(float(box.xyxy[0][2]), 1),
                "y2": round(float(box.xyxy[0][3]), 1),
            },
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">♻️ EcoAudit</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered waste detection — upload an image to identify waste types</div>', unsafe_allow_html=True)
st.divider()

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Lower = more detections (including uncertain ones). Higher = only confident detections.",
    )
    st.divider()
    st.markdown("**Detectable waste types:**")
    for cls, color in CLASS_COLORS.items():
        st.markdown(
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{color};border-radius:3px;margin-right:6px;"></span>{cls}',
            unsafe_allow_html=True,
        )

# ── Upload ──────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color: #aaa;">
        <div style="font-size:3rem;">📂</div>
        <div style="font-size:1.1rem; margin-top:0.5rem;">Drag and drop an image here, or click <b>Browse files</b></div>
        <div style="font-size:0.85rem; margin-top:0.3rem;">Supports JPG, PNG, WEBP</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Detection ───────────────────────────────────────────────────────────────────
model = load_model()
image = Image.open(uploaded_file).convert("RGB")

with st.spinner("Detecting waste..."):
    result = run_detection(model, image, conf_threshold)
    detections = build_detections(result)
    annotated_image = get_annotated_image(result)

# ── Results layout ───────────────────────────────────────────────────────────────
col_img, col_results = st.columns([3, 2], gap="large")

with col_img:
    tab_annotated, tab_original = st.tabs(["📍 Annotated", "🖼️ Original"])
    with tab_annotated:
        st.image(annotated_image, use_container_width=True)
    with tab_original:
        st.image(image, use_container_width=True)

with col_results:
    count = len(detections)
    if count == 0:
        st.warning(f"No waste detected above {conf_threshold:.0%} confidence. Try lowering the threshold in the sidebar.")
    else:
        st.markdown(f'<div class="section-label">Detected — {count} item{"s" if count != 1 else ""}</div>', unsafe_allow_html=True)

        for i, det in enumerate(detections):
            color = det["color"]
            conf_pct = int(det["confidence"] * 100)
            bbox = det["bbox"]
            st.markdown(f"""
            <div class="detection-card" style="border-color:{color};">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <span style="font-weight:600;font-size:1rem;">
                        <span style="display:inline-block;width:12px;height:12px;background:{color};
                        border-radius:3px;margin-right:6px;vertical-align:middle;"></span>
                        {det["class_name"]}
                    </span>
                    <span style="font-weight:700;color:{color};">{conf_pct}%</span>
                </div>
                <div class="conf-bar-bg">
                    <div style="background:{color};height:8px;border-radius:999px;width:{conf_pct}%;"></div>
                </div>
                <div style="font-size:0.75rem;color:#999;margin-top:6px;">
                    Box: ({bbox["x1"]}, {bbox["y1"]}) → ({bbox["x2"]}, {bbox["y2"]})
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Download annotated image
        buf = io.BytesIO()
        annotated_image.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download annotated image",
            data=buf.getvalue(),
            file_name=f"ecoaudit_{uploaded_file.name}",
            mime="image/png",
            use_container_width=True,
        )
