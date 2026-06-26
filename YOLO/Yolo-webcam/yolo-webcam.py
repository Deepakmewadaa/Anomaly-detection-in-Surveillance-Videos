from ultralytics import YOLO
import cv2
import math

# For webcam

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# For Video

# cap = cv2.VideoCapture("video-cars.mp4")

model = YOLO('../Yolo-weights/yolov8n.pt')
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,255), 2)

            conf = math.ceil(box.conf[0]*100)/100
            

            # Class Name

            c = int(box.cls)
            cv2.putText(img, str(conf)+" " + str(r.names[c]), (x1,y1), 2, 1.5, (0,0,255), 1 )
    
    cv2.imshow("Live", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
