from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")  # downloads automatically on first run

    model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,       # 0 = first GPU, "cpu" = force CPU
        workers=2,      # reduce to avoid paging file error on Windows
        project="runs",
        name="ecoaudit_v1",
    )
