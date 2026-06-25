import cv2
from ultralytics import YOLO

model = YOLO('../Yolo-weights/yolov5n.pt')
result = model('cars.jpg')

img = result[0].plot()
cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
