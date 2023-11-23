from ultralytics import YOLO
import glob
import os
import cv2
import supervision as sv
import json
import numpy as np

os.makedirs('/home/praveenvnktsh/alfred-autonomy/src/manipulation/scripts/sim/images/detections', exist_ok=True)
os.makedirs('/home/praveenvnktsh/alfred-autonomy/src/manipulation/scripts/sim/images/detection_img', exist_ok=True)
# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)


for i, path in enumerate(sorted(glob.glob('/home/praveenvnktsh/alfred-autonomy/src/manipulation/scripts/sim/images/rgb/*.png'))):
    print(path)
    img = cv2.imread(path)  # BGR
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    results = model(img, show = False,)
    boxes =  results[0].boxes
    detections = sv.Detections.from_ultralytics(results[0])
    confs, box_cls, box = detections.confidence, detections.class_id, detections.xyxy
    final_results = [
        [],
        [],
        [],
        []
    ]
    h, w = img.shape[:2]
    upscale_fac = 1.0
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    for (b, cls, conf) in zip(box,box_cls, confs):
        x1, y1, x2, y2 = b
        
        # apply correct transform to rotate box back
        y_new1 = w - x2 * upscale_fac
        x_new1= y1 * upscale_fac
        y_new2 = w - x1 * upscale_fac
        x_new2 = y2 * upscale_fac
        
        b = [x_new1, y_new1, x_new2, y_new2]
        final_results[0].append(b)
        final_results[1].append(cls) #remapped class index
        final_results[2].append(conf)
        img = cv2.rectangle(img, (int(b[0]),int(b[1])),(int(b[2]),int(b[3])) , (255,0,0), 2)
        
        img = cv2.putText(img, str(cls) ,(int(b[0]),int(b[1])),  cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0,0,255), 1, cv2.LINE_AA)
        
    cv2.imwrite('/home/praveenvnktsh/alfred-autonomy/src/manipulation/scripts/sim/images/detection_img/' + str(i).zfill(6) + '.png' , img)
    with open('/home/praveenvnktsh/alfred-autonomy/src/manipulation/scripts/sim/images/detections/' + str(i).zfill(6)  + '.json', 'w') as f:
        dic = {
            'boxes': np.array(final_results[0]).tolist(),
            'classes': np.array(final_results[1]).tolist(),
            'scores': np.array(final_results[2]).tolist(),
        }
        json.dump(dic, f)