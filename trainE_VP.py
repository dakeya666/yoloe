from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPETrainer,YOLOETrainer,YOLOETrainerFromScratch,YOLOEVPTrainer


# Initialize a detection model from a config
model = YOLOE("yoloe-26s.yaml")
#model.load("./runs/detect/ybj/weights/best.pt")
# Load weights from a pretrained segmentation checkpoint (same scale)
model.load("./runs/detect/ybj0310/weights/best.pt")
#
data = dict(
        train = dict(yolo_data=["ybj20260303-yoloe.yaml"]),
        val = dict(yolo_data=["ybj20260303-yoloe.yaml"])
       # train = dict(yolo_data=["coco128.yaml"]),
       # val = dict(yolo_data=["coco128.yaml"])
        )

head_index = len(model.model.model) - 1
freeze = list(range(0, head_index))
for name, child in model.model.model[-1].named_children():
    if "savpe" not in name:
        freeze.append(f"{head_index}.{name}")
# Fine-tune on your detection dataset
results = model.train(
  #  cfg ="cloth_small.yaml", 
    data=data,  # Detection dataset
    batch = 8,
    optimizer = 'AdamW',
    lr0=0.0005,
    warmup_bias_lr = 0.0,
    weight_decay=0.025,
    momentum = 0.9,
    name = "VP",
    save_period = 10,
    epochs =20,
    trainer = YOLOEVPTrainer,
    device = 1,
    freeze=freeze,
)
