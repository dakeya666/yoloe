from ultralytics import YOLOE
from ultralytics.utils.patches import torch_load
import numpy as np
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
#model = YOLOE("yoloe-26s-seg.pt")
model = YOLOE("./runs/detect/VP/weights/best.pt")


#det_model = YOLOE("yoloe-26s.yaml")
#state = torch_load("yoloe-26s-seg.pt")
#det_model.load((state["model"]))
#det_model.save("yoloe-26s-det.pt")
#model.set_classes(["pinhole"])
visual_prompts = dict(
    bboxes=np.array([[383, 620, 469, 744]]),
    cls=np.array([0]),
)
# Run inference on multiple images, using the provided visual prompts as guidance
results = model.predict(
    "./testImg",
    conf = 0.2,
    #refer_image = ["./testImg/1.jpg"],
    #visual_prompts=visual_prompts,
    save=True
)

