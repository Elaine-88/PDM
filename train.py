
from ultralytics import YOLO


if __name__ == "__main__":
    # Load a model
    model = YOLO('yolov8x-pose.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data='./Tooth.yaml',
                          epochs=2,
                          imgsz=640,
                          batch=4,
                          device=0,
                          save_period=200,
                          # workers=128,
                          cos_lr=True,
                          patience=1200)
