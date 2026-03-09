from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("runs/detect/runs/ecoaudit_v17/weights/best.pt")

    model.val(
        data="dataset/data.yaml",
        split="test",
        project="runs",
        name="ecoaudit_v17_test",
    )
