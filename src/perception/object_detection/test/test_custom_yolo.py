from ultralytics import YOLO
import cv2
model = YOLO('best.pt')
model.to('cuda:0')
img = cv2.imread('temp.png')
results = model(img)
boxes =  results[0].boxes
box = boxes.xyxy
box_cls =boxes.cls

for (b,cls) in zip(box,box_cls):
    img = cv2.rectangle(img, (int(b[0]),int(b[1])),(int(b[2]),int(b[3])) , (255,0,0), 2)
    
    img = cv2.putText(img, str(model.names[cls.item()]) + " | " + str(cls.item()),(int(b[0]),int(b[1])),  cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0,0,255), 1, cv2.LINE_AA)

# cv2.imshow('img', img)
# cv2.waitKey(0)

cv2.imwrite('temp.png', img)