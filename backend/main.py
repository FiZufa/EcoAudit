import io
from pathlib import Path
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, ImageDraw, ImageFont

MODEL_PATH = Path(__file__).parent.parent / "training/runs/detect/runs/ecoaudit_v17/weights/best.pt"

app = FastAPI(title="EcoAudit Waste Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO(str(MODEL_PATH))

# Distinct colors per class (RGB)
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
