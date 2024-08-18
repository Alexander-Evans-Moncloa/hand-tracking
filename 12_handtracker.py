import cv2
import mediapipe as mp
import time
import handtrackingmodule as htm

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img =cap.read()
    img = detector.findHands(img, draw=True)           #replace (img, draw=True) with (img, draw = False) for no lines or automatic red dots
    lmList = detector.findPosition(img, draw=True)    #same as above to remove blue dots
    if len(lmList) != 0:
        print(lmList [0])

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 2, (225,0,0), 2)

    cv2.imshow('handicam.com', img)
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break 