from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # downloads automatically on first run

model.train(
    data="dataset_yolo/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,       # 0 = first GPU, "cpu" = force CPU
    project="runs",
    name="ecoaudit_v1",
)
