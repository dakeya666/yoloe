from ultralytics import YOLOE
from ultralytics.utils.patches import torch_load
import numpy as np
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
#model = YOLOE("yoloe-26s-seg.pt")
model = YOLOE("./runs/detect/ybj/weights/best.pt")


#det_model = YOLOE("yoloe-26s.yaml")
#state = torch_load("yoloe-26s-seg.pt")
#det_model.load((state["model"]))
#det_model.save("yoloe-26s-det.pt")
#model.set_classes(["pinhole"])
#visual_prompts = dict(
#    bboxes=np.array([[377, 615, 471, 735]]),
#    cls=np.array([0]),
#)
data = dict(
        train = dict(yolo_data=["ybj20260303-yoloe.yaml"]),
        val = dict(yolo_data=["ybj20260303-yoloe.yaml"])
        #train = dict(yolo_data=["VOC.yaml"]),
        #val = dict(yolo_data=["VOC.yaml"])
        )


# Run inference on multiple images, using the provided visual prompts as guidance
results = model.val(
    data = "ybj20260303-yoloe.yaml",
    #refer_image = ["./testImg/1.jpg"],
    #visual_prompts=visual_prompts,
    conf = 0.3,
    save=True
)


