import cv2
import mediapipe as mp
import mediapipe.tasks.python as tasks
import mediapipe.tasks.python.vision as handStuff

#select hand solution
handSource = mp.solutions.hands
hands = handSource.Hands()

#open camera -- 0 is default
videoCap = cv2.VideoCapture(0)
#read image
success, img = videoCap.read()
#show image on separate window
if success: 
    #recognize hands from image
    recognizeHands = hands.process(img)
    if recognizeHands.multi_hand_landmarks: 
        #draw the dots on each image
        for datapoint, point in enumerate(hand.landmark): 
            h, w, c = img.shape
            x, y = int(point.x * w), int(point.y * h)
            cv2.circle(img, (x,y), 10, (255, 0, 255), cv2.FILLED)
            cv2.imshow("CamOutput", img)
            cv2.waitKey(1)