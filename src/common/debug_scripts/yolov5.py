#!/usr/local/lib/obj_det_env/bin/python3

import torch
# import supervision as sv
# Model
import time
from ultralytics import YOLO
model = YOLO('yolov8x.pt')
model = YOLO('yolov8x.pt')

# Images
img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
starttime = time.time()
for i in range(100):
    result = model(img, show = False, verbose= False)[0]
    print(result)
    
print(100.0/(time.time() - starttime))
    # detections = sv.Detections.from_yolov8(result)    
    # print(detections.confidence, detections.class_id, )
for i in range(10):
    result = model(img, show = False, verbose= False)[0]
    print(result)

    # detections = sv.Detections.from_yolov8(result)
    # print(detections.confidence, detections.class_id, )
# labels = [
#     f"{model.model.names[class_id]} {confidence:0.2f}"
#     for _, confidence, class_id, _
#     in detections
# ]
