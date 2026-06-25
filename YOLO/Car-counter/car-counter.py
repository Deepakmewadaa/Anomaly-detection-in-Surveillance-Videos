from ultralytics import YOLO
import cv2
import math

cap = cv2.VideoCapture("cars.mp4")
mask = cv2.imread("mask.png")
model = YOLO('../Yolo-weights/yolov8n.pt')
while True:
    success, img = cap.read()
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))

    imgRegion = cv2.bitwise_and(img, mask_resized)
    results = model(imgRegion, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            

            conf = math.ceil(box.conf[0]*100)/100
            

            # Class Name

            c = int(box.cls)
            class_name = r.names[c]
            num_cars = 0
            if class_name == "car" or class_name=="truck" or class_name=="bus" or class_name=="motorbike":
                num_cars += 1
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,255), 2)
                cv2.putText(img, str(conf)+" " + str(class_name), (x1,y1), 2, 1.5, (0,0,255), 1 )
    
    cv2.imshow("Live", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
