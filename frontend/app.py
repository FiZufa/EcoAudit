import json
from pathlib import Path
import io
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent.parent / "training/runs/detect/runs/ecoaudit_v17/weights/best.pt"
CONFIG_PATH = Path(__file__).parent / "diagnostic_config.json"

with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

# Build label → tier key mapping
LABEL_TO_TIER: dict[str, str] = {}
for tier_key, tier_data in CONFIG["categories"].items():
    for label in tier_data["labels"]:
        LABEL_TO_TIER[label] = tier_key

DISPOSAL_GUIDANCE: dict[str, str] = CONFIG["disposal_guidance"]
SCORING = CONFIG["scoring_logic"]
TIER_ORDER = list(CONFIG["categories"].keys())  # tier_1 … tier_4


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _badge_text_color(hex_color: str) -> str:
    """Black text on bright yellow/green tiles, white otherwise."""
    return "#333" if hex_color in ("#FFFF00", "#00FF00") else "#fff"


def compute_purity_score(detections: list[dict]) -> int:
    score = SCORING["base_score"]
    if SCORING["tier_1_behavior"] == "immediate_zero":
        for det in detections:
            if det["tier"] == "tier_1":
                return SCORING["min_score"]
    for det in detections:
        score -= CONFIG["categories"][det["tier"]]["penalty_per_unit"]
    return max(score, SCORING["min_score"])


# ── Streamlit config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoAudit — Waste Diagnostic",
    page_icon="♻️",
    layout="wide",
)

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #2E7D32; }
    .subtitle   { color: #666; margin-bottom: 1.5rem; }
    .score-number { font-size: 3.5rem; font-weight: 800; line-height: 1; }
    .status-badge {
        display: inline-block;
        padding: 6px 20px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.06em;
    }
    .status-clean         { background:#E8F5E9; color:#2E7D32; border:2px solid #2E7D32; }
    .status-contaminated  { background:#FFEBEE; color:#C62828; border:2px solid #C62828; }
    .detection-card {
        background: #f8f9fa;
        border-left: 4px solid;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .tier-badge {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 2px 7px;
        border-radius: 999px;
    }
    .guidance-text { font-size: 0.82rem; color: #555; margin-top: 6px; line-height: 1.45; }
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


def build_detections(result) -> list[dict]:
    detections = []
    for box in result.boxes:
        cls_name = result.names[int(box.cls)]
        tier = LABEL_TO_TIER.get(cls_name, "tier_4")
        tier_data = CONFIG["categories"][tier]
        detections.append({
            "class_name": cls_name,
            "confidence": round(float(box.conf), 4),
            "tier": tier,
            "tier_color": tier_data["color_code"],
            "tier_name": tier_data["name"],
            "severity": tier_data["severity"],
            "guidance": DISPOSAL_GUIDANCE.get(cls_name, "Follow local waste disposal guidelines."),
            "bbox": {
                "x1": round(float(box.xyxy[0][0]), 1),
                "y1": round(float(box.xyxy[0][1]), 1),
                "x2": round(float(box.xyxy[0][2]), 1),
                "y2": round(float(box.xyxy[0][3]), 1),
            },
        })
    # Sort by tier severity (tier_1 first)
    detections.sort(key=lambda d: TIER_ORDER.index(d["tier"]))
    return detections


def draw_diagnostic_boxes(image: Image.Image, detections: list[dict]) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except OSError:
        font = ImageFont.load_default()

    for det in detections:
        color_rgb = hex_to_rgb(det["tier_color"])
        b = det["bbox"]
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        conf_pct = int(det["confidence"] * 100)
        label = f"{det['class_name']} {conf_pct}%"

        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=3)

        tb = draw.textbbox((x1, y1), label, font=font)
        text_w, text_h = tb[2] - tb[0], tb[3] - tb[1]
        label_y = y1 - text_h - 6 if y1 - text_h - 6 > 0 else y1 + 2

        draw.rectangle([x1, label_y, x1 + text_w + 6, label_y + text_h + 4], fill=color_rgb)
        text_fill = "black" if det["tier_color"] in ("#FFFF00", "#00FF00") else "white"
        draw.text((x1 + 3, label_y + 2), label, fill=text_fill, font=font)

    return img


# ── Header ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">♻️ EcoAudit</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered waste diagnostic — upload an image to assess purity and get disposal guidance</div>', unsafe_allow_html=True)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.1, max_value=1.0, value=0.5, step=0.05,
        help="Lower = more detections. Higher = only confident detections.",
    )
    st.divider()
    st.markdown("**Tier legend:**")
    for tier_key, tier_data in CONFIG["categories"].items():
        color = tier_data["color_code"]
        tc = _badge_text_color(color)
        st.markdown(
            f'<span class="tier-badge" style="background:{color};color:{tc};">'
            f'{tier_data["severity"]}</span>&nbsp; <b>{tier_data["name"]}</b>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<span style="font-size:0.75rem;color:#888;">'
            f'{", ".join(tier_data["labels"])}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("")

# ── Upload ────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color: #aaa;">
        <div style="font-size:3rem;">📂</div>
        <div style="font-size:1.1rem; margin-top:0.5rem;">
            Drag and drop an image here, or click <b>Browse files</b>
        </div>
        <div style="font-size:0.85rem; margin-top:0.3rem;">Supports JPG, PNG, WEBP</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Run diagnostic ────────────────────────────────────────────────────────────────
model = load_model()
image = Image.open(uploaded_file).convert("RGB")

with st.spinner("Running diagnostic..."):
    result = run_detection(model, image, conf_threshold)
    detections = build_detections(result)
    purity_score = compute_purity_score(detections)
    status = "CLEAN" if purity_score == 100 else "CONTAMINATED"
    annotated_image = draw_diagnostic_boxes(image, detections)

# ── Layout ────────────────────────────────────────────────────────────────────────
col_img, col_results = st.columns([3, 2], gap="large")

with col_img:
    tab_annotated, tab_original = st.tabs(["📍 Annotated", "🖼️ Original"])
    with tab_annotated:
        st.image(annotated_image, use_container_width=True)
    with tab_original:
        st.image(image, use_container_width=True)

with col_results:
    # ── Purity score & status ────────────────────────────────────────────────────
    score_color = (
        "#2E7D32" if purity_score >= 80
        else "#F57F17" if purity_score >= 50
        else "#C62828"
    )
    status_css = "status-clean" if status == "CLEAN" else "status-contaminated"

    st.markdown(f"""
    <div style="text-align:center;padding:1.5rem;background:#f8f9fa;border-radius:12px;margin-bottom:0.8rem;">
        <div class="section-label">Purity Score</div>
        <div class="score-number" style="color:{score_color};">
            {purity_score}<span style="font-size:1.5rem;">%</span>
        </div>
        <div style="margin-top:0.8rem;">
            <span class="status-badge {status_css}">{status}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#e0e0e0;border-radius:999px;height:10px;margin-bottom:1.2rem;">
        <div style="background:{score_color};height:10px;border-radius:999px;width:{purity_score}%;"></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Detection cards ──────────────────────────────────────────────────────────
    count = len(detections)
    if count == 0:
        st.warning(f"No waste detected above {conf_threshold:.0%} confidence. Try lowering the threshold.")
    else:
        st.markdown(
            f'<div class="section-label">Detected — {count} item{"s" if count != 1 else ""}</div>',
            unsafe_allow_html=True,
        )
        for det in detections:
            tc = det["tier_color"]
            btc = _badge_text_color(tc)
            conf_pct = int(det["confidence"] * 100)
            st.markdown(f"""
            <div class="detection-card" style="border-color:{tc};">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                    <span style="font-weight:600;font-size:1rem;">{det["class_name"]}</span>
                    <span class="tier-badge" style="background:{tc};color:{btc};">{det["severity"]}</span>
                </div>
                <div style="font-size:0.8rem;color:#888;margin-bottom:4px;">
                    Confidence: <b>{conf_pct}%</b> &nbsp;|&nbsp; {det["tier_name"]}
                </div>
                <div class="guidance-text">📋 {det["guidance"]}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Global handling instructions ─────────────────────────────────────────
        tiers_found = sorted(
            set(d["tier"] for d in detections),
            key=lambda t: TIER_ORDER.index(t),
        )
        st.divider()
        st.markdown('<div class="section-label">Handling Instructions</div>', unsafe_allow_html=True)
        for tier in tiers_found:
            td = CONFIG["categories"][tier]
            tc = td["color_code"]
            btc = _badge_text_color(tc)
            st.markdown(f"""
            <div style="background:#f8f9fa;border-left:4px solid {tc};border-radius:6px;
                        padding:8px 12px;margin-bottom:8px;">
                <span class="tier-badge" style="background:{tc};color:{btc};margin-right:8px;">
                    {td["severity"]}
                </span>
                <span style="font-size:0.85rem;">{td["global_instruction"]}</span>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    buf = io.BytesIO()
    annotated_image.save(buf, format="PNG")
    st.download_button(
        label="⬇️ Download diagnostic image",
        data=buf.getvalue(),
        file_name=f"ecoaudit_diagnostic_{uploaded_file.name}",
        mime="image/png",
        use_container_width=True,
    )
