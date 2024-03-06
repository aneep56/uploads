import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import pyfirmata

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 360)

detector = HandDetector(detectionCon=0.8, maxHands=2)  # Allowing for detection of two hands

minHand, maxHand = 20, 250
minBar, maxBar = 400, 150
minAngle, maxAngle = 0, 180

port = "COM3"
board = pyfirmata.Arduino(port)
servoPinLeft = board.get_pin('d:12:s')  # Pin 12 Arduino for left hand
servoPinRight = board.get_pin('d:13:s')  # Pin 13 Arduino for right hand


while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands and len(hands) == 2:  # If two hands are detected
        # Left Hand
        thumbTipLeft = hands[0]["lmList"][4][0:2]
        indexTipLeft = hands[0]["lmList"][8][0:2]
        lengthLeft, _, _ = detector.findDistance(indexTipLeft, thumbTipLeft, img)

        # Right Hand
        thumbTipRight = hands[1]["lmList"][4][0:2]
        indexTipRight = hands[1]["lmList"][8][0:2]
        lengthRight, _, _ = detector.findDistance(indexTipRight, thumbTipRight, img)

        # Interpolating distances to servo values
        servoValLeft = np.interp(lengthLeft, [minHand, maxHand], [minAngle, maxAngle])
        servoValRight = np.interp(lengthRight, [minHand, maxHand], [minAngle, maxAngle])

        # Display length and servo values
        cv2.putText(img, f'Left Length: {int(lengthLeft)}', (30, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 255, 255), 3)
        cv2.putText(img, f'Right Length: {int(lengthRight)}', (30, 100), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 255, 255), 3)

        cv2.putText(img, f'Left Servo: {int(servoValLeft)}', (30, 150), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 255, 255), 3)
        cv2.putText(img, f'Right Servo: {int(servoValRight)}', (30, 200), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 255, 255), 3)

        # Writing servo values to the respective pins
        servoPinLeft.write(servoValLeft)
        servoPinRight.write(servoValRight)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
