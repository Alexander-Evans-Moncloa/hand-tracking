import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5): #Defines function that allows self object to have varying parameters
        self.mode = mode        #Creating an object with its own variable (self. something is the variable of the object) assigning it a value by the user
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):                   #Defines function to find hands
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)         #Actually processes the image using the hands object
        #print(results.multi_hand_landmarks)     #Prints results (either None or landmarks with coordinates)

        if self.results.multi_hand_landmarks:        #If there is a result
            for handLms in self.results.multi_hand_landmarks:    #for a number going upwards (calling it handLms instead of i) up to the value of the result
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    #Draws DOTS on the original image using mpDraw.draw_landmarks function, THEN lines with mpHands.HAND_CONNECTIONS
        
        return img   

    def findPosition(self, img, handNo=0, draw = True):
        
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):     #Where get the landmark information (x and y coordinates) and ID numbers (which finger joint is which)
                h, w, c = img.shape                     #Gets height, width and channels of image
                cx, cy = int(lm.x*w), int(lm.y*h)       #calculates position off the centre (int changes it from decimal places)
                lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList
        

def main():
    pTime = 0       #Initialises variables
    cTime = 0
    cap = cv2.VideoCapture(0)

    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1/(cTime-pTime)       #Calculates frames per second
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (225, 0, 0), 3)       #Displays them in the corner (in blue)

        cv2.imshow('handicam.com', img)
        if cv2.waitKey(1) & 0xFF == ord('p'):      #Shows images at the camera FPS and if someone presses the letter "p" then it will pause the webcam stream.
            break 

if __name__ == "__main__":
    main()

