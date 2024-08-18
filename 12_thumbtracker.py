import cv2
import mediapipe as mp
import time

pTime = 0       #Initialises variables
cTime = 0

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands                #Imports hands ML database from mediapipe

hands = mpHands.Hands()                     #Gets hand from mpHands object

mpDraw = mp.solutions.drawing_utils         #Function that draws the lines between dots for us (very nifty)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)         #Actually processes the image using the hands object
    #print(results.multi_hand_landmarks)     #Prints results (either None or landmarks with coordinates)

    if results.multi_hand_landmarks:        #If there is a result
        for handLms in results.multi_hand_landmarks:    #for a number going upwards (calling it handLms instead of i) up to the value of the result
            for id, lm in enumerate(handLms.landmark):     #Where get the landmark information (x and y coordinates) and ID numbers (which finger joint is which)
                #print(id,lm)
                h, w, c = img.shape                     #Gets height, width and channels of image
                cx, cy = int(lm.x*w), int(lm.y*h)       #calculates position off the centre (int changes it from decimal places)
                print(id, cx,cy)
                if id == 4:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)      #Finds thumb and makes the dot larger

            #mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  #Draw DOTS on the original image using mpDraw.draw_landmarks function, THEN lines with mpHands.HAND_CONNECTIONS
    

    cTime = time.time()
    fps = 1/(cTime-pTime)       #Calculates frames per second
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (225, 0, 0), 3)       #Displays them in the corner (in blue)

    cv2.imshow('thumbcam', img)
    if cv2.waitKey(1) & 0xFF == ord('p'):      #Shows images at the camera FPS and if someone presses the letter "p" then it will pause the webcam stream.
        break 

cap.release()
cv2.destroyAllWindows()
