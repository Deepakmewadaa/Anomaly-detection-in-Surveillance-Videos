import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = face_cascade.detectMultiScale(grey, 1.1, 5)
    
    """
    detectMultiScale() -> Scan and Detect faces
    1.1 -> Scale Factor
    5 -> minNeighbours 
    """

    for (x, y, w, h) in face:
        cv2.rectangle(frame, pt1=(x,y), pt2=(x+w, y+h), color=(0,0,255), thickness=2)

    roi_grey = grey[y:y+h, x:x+w]

    eye = eye_cascade.detectMultiScale(roi_grey, 1.1, 10)
    if len(eye)>0:
        cv2.putText(frame, "Eye Detected", (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    
    smile = smile_cascade.detectMultiScale(roi_grey, 1.1, 10)
    if len(smile)>0:
        cv2.putText(frame, "Smile :-)", (x+200, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    
    cv2.imshow("Live Face Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()