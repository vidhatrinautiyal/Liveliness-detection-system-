from ultralytics import YOLO

model = YOLO('models/yolov8n.pt')


def main():
    # defined to perform model training.
    # Reduce the batch size
    # Set your desired batch size
    model.train(data='Dataset/SplitData/dataOffline.yaml', epochs=50, imgsz=640, batch=2, device='cpu', workers=2)

if __name__ == '__main__':
    main()
