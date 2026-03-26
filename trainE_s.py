from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPETrainer,YOLOETrainer,YOLOETrainerFromScratch,YOLOEVPTrainer


# Initialize a detection model from a config
model = YOLOE("yoloe-26s.yaml")
model.load("yoloe-26s-seg.pt")
# Load weights from a pretrained segmentation checkpoint (same scale)

data = dict(
        #train = dict(yolo_data=["ybj20260303-yoloe.yaml"]),
        #val = dict(yolo_data=["ybj20260303-yoloe.yaml"])
        train = dict(yolo_data=["coco128.yaml"]),
        val = dict(yolo_data=["coco128.yaml"])
        )

# Fine-tune on your detection dataset
results = model.train(
    data=data,  # Detection dataset
    name = "coco",
    epochs=50,
    trainer = YOLOETrainerFromScratch,
    device = 1
)
