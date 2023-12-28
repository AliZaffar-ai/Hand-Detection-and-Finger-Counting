import cv2
import mediapipe as mp
import time
import numpy as np

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands)
        self.mpDraw = mp.solutions.drawing_utils

   


    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        handType = None
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            # Detect hand type (left or right)
            if self.results.multi_handedness:
                handType = self.results.multi_handedness[handNo].classification[0].label
    
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 8, (162,126,64), cv2.FILLED)
        return lmList, handType



    def countFingers(self, img, lmList, handType):
        if not lmList:
           return 0

        fingerTips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky finger tips
        count = 0

    # Thumb: Using distance between thumb tip and index finger base to determine if the thumb is extended
        thumbTip = lmList[fingerTips[0]]
        indexFingerBase = lmList[fingerTips[1] - 2]
        thumbIsOpen = np.linalg.norm(np.array(thumbTip[1:]) - np.array(indexFingerBase[1:])) > np.linalg.norm(np.array(lmList[2][1:]) - np.array(lmList[3][1:]))

        if thumbIsOpen:
            count += 1

    # Fingers 2-5 (Index to Pinky)
        for i in range(1, 5):
            if lmList[fingerTips[i]][2] < lmList[fingerTips[i] - 2][2]:
               count += 1

        cv2.putText(img, f'Fingers: {count}', (10, 140), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
        return count



def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, handType = detector.findPosition(img)  # Updated to receive hand type

        if len(lmList) != 0:
            detector.countFingers(img, lmList, handType)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 255, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
