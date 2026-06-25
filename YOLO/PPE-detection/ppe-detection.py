from ultralytics import YOLO
import cv2
import math

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO('PPE-detection/best.pt')

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            

            conf = math.ceil(box.conf[0]*100)/100
            

            # Class Name

            c = int(box.cls)
            class_name = r.names[c]
            if class_name == "NO-Safety Vest" or class_name=="NO-Mask" or class_name=="NO-Hardhat":
                curr_color = (0,0,255)
            else:
                curr_color = (0,128,0)
            
            cv2.putText(img, str(conf)+" " + str(r.names[c]), (x1,y1), 2, 1.5, curr_color, 1 )
            cv2.rectangle(img, (x1, y1), (x2, y2), curr_color, 2)
            
    cv2.imshow("Live", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
