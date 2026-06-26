from ultralytics import YOLO
import cv2
import math
from sort import *

cap = cv2.VideoCapture("YOLO\cars.mp4")
mask = cv2.imread("YOLO\mask.png")
model = YOLO('../Yolo-weights/yolov8n.pt')

tractor = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [400, 297, 673, 297]
total_counts = []


while True:
    success, img = cap.read()
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))

    imgRegion = cv2.bitwise_and(img, mask_resized)
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = math.ceil(box.conf[0]*100)/100
            

            # Class Name

            c = int(box.cls)
            class_name = r.names[c]
            if class_name == "car" or class_name=="truck" or class_name=="bus" or class_name=="motorbike" and conf > 0.3:

                # cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,255), 2)
                # cv2.putText(img, str(conf)+" " + str(class_name), (x1,y1), 1, 1, (0,0,255), 1)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))

    tractor_result = tractor.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in tractor_result:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2-x1, y2-y1
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,255), 2)
        cv2.putText(img, str(id) , (x1,y1), 1, 1, (0,0,255), 1)

        cx, cy = x1 + w//2, y1+h//2
        cv2.circle(img, (cx,cy), 5, (255,0,255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1]-15<cy<limits[3]:
            if id not in total_counts:
                total_counts.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    cv2.putText(img, str(len(total_counts)), (50,70), 3, 3, (255,0,0), 4)
    cv2.imshow("Live", img)
    # cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
