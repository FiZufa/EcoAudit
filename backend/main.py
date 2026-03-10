import io
import json
from pathlib import Path
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, ImageDraw, ImageFont

MODEL_PATH = Path(__file__).parent.parent / "training/runs/detect/runs/ecoaudit_v17/weights/best.pt"
CONFIG_PATH = Path(__file__).parent.parent / "frontend/diagnostic_config.json"

app = FastAPI(title="EcoAudit Waste Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO(str(MODEL_PATH))

with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

LABEL_TO_TIER: dict[str, str] = {}
for _tier_key, _tier_data in CONFIG["categories"].items():
    for _label in _tier_data["labels"]:
        LABEL_TO_TIER[_label] = _tier_key

SCORING = CONFIG["scoring_logic"]
TIER_ORDER = list(CONFIG["categories"].keys())

# Distinct colors per class for legacy /detect endpoints (RGB)
CLASS_COLORS = [
    (255,  99,  71),  # Battery       - tomato red
    ( 60, 179, 113),  # Biological    - medium sea green
    ( 70, 130, 180),  # Cable         - steel blue
    (255, 165,   0),  # Can           - orange
    (139,  90,  43),  # Cardboard     - brown
    (218, 112, 214),  # Clothes       - orchid
    (255, 215,   0),  # E-waste       - gold
    (100, 149, 237),  # Glass         - cornflower blue
    (220,  20,  60),  # Medical Waste - crimson
    (169, 169, 169),  # Metal         - dark gray
    (144, 238, 144),  # Paper         - light green
    ( 30, 144, 255),  # Plastic       - dodger blue
    (255, 182, 193),  # Shoes         - light pink
    (128,   0, 128),  # Trash         - purple
]


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _load_image(contents: bytes) -> Image.Image:
    return Image.open(io.BytesIO(contents)).convert("RGB")


def _run_detection(image: Image.Image, conf: float):
    results = model.predict(image, conf=conf, verbose=False)
    return results[0]


def _build_detections(result) -> list[dict]:
    detections = []
    for box in result.boxes:
        detections.append({
            "class_id": int(box.cls),
            "class_name": model.names[int(box.cls)],
            "confidence": round(float(box.conf), 4),
            "bbox": {
                "x1": round(float(box.xyxy[0][0]), 2),
                "y1": round(float(box.xyxy[0][1]), 2),
                "x2": round(float(box.xyxy[0][2]), 2),
                "y2": round(float(box.xyxy[0][3]), 2),
            },
        })
    return detections


def _build_diagnostic_detections(result) -> list[dict]:
    detections = []
    for box in result.boxes:
        cls_name = model.names[int(box.cls)]
        tier = LABEL_TO_TIER.get(cls_name, "tier_4")
        tier_data = CONFIG["categories"][tier]
        detections.append({
            "class_name": cls_name,
            "confidence": round(float(box.conf), 4),
            "tier": tier,
            "tier_name": tier_data["name"],
            "severity": tier_data["severity"],
            "tier_color": tier_data["color_code"],
            "guidance": CONFIG["disposal_guidance"].get(
                cls_name, "Follow local waste disposal guidelines."
            ),
            "bbox": {
                "x1": round(float(box.xyxy[0][0]), 2),
                "y1": round(float(box.xyxy[0][1]), 2),
                "x2": round(float(box.xyxy[0][2]), 2),
                "y2": round(float(box.xyxy[0][3]), 2),
            },
        })
    detections.sort(key=lambda d: TIER_ORDER.index(d["tier"]))
    return detections


def _compute_purity_score(detections: list[dict]) -> int:
    score = SCORING["base_score"]
    if SCORING["tier_1_behavior"] == "immediate_zero":
        for det in detections:
            if det["tier"] == "tier_1":
                return SCORING["min_score"]
    for det in detections:
        score -= CONFIG["categories"][det["tier"]]["penalty_per_unit"]
    return max(score, SCORING["min_score"])


def _draw_boxes(image: Image.Image, result) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except OSError:
        font = ImageFont.load_default()

    for box in result.boxes:
        cls_id = int(box.cls)
        label = f"{model.names[cls_id]} {float(box.conf):.2f}"
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        label_y = y1 - text_h - 6 if y1 - text_h - 6 > 0 else y1 + 2
        draw.rectangle([x1, label_y, x1 + text_w + 6, label_y + text_h + 4], fill=color)
        draw.text((x1 + 3, label_y + 2), label, fill="white", font=font)

    return image


def _draw_diagnostic_boxes(image: Image.Image, detections: list[dict]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except OSError:
        font = ImageFont.load_default()

    for det in detections:
        color_rgb = _hex_to_rgb(det["tier_color"])
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

    return image


# ── Routes ────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "EcoAudit Waste Detection API", "status": "running"}


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = 0.5):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    image = _load_image(contents)
    result = _run_detection(image, conf)

    return JSONResponse({
        "filename": file.filename,
        "detections": _build_detections(result),
        "count": len(result.boxes),
    })


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...), conf: float = 0.5):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    image = _load_image(contents)
    result = _run_detection(image, conf)
    annotated = _draw_boxes(image, result)

    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@app.post("/diagnostic")
async def diagnostic(file: UploadFile = File(...), conf: float = 0.5):
    """Return purity score, status, and per-detection disposal guidance."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    image = _load_image(contents)
    result = _run_detection(image, conf)
    detections = _build_diagnostic_detections(result)
    purity_score = _compute_purity_score(detections)
    status = "CLEAN" if purity_score == 100 else "CONTAMINATED"

    return JSONResponse({
        "filename": file.filename,
        "purity_score": purity_score,
        "status": status,
        "count": len(detections),
        "detections": detections,
    })


@app.post("/diagnostic/image")
async def diagnostic_image(file: UploadFile = File(...), conf: float = 0.5):
    """Return the image annotated with tier-colored bounding boxes."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    image = _load_image(contents)
    result = _run_detection(image, conf)
    detections = _build_diagnostic_detections(result)
    annotated = _draw_diagnostic_boxes(image, detections)

    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
