import cv2
import time
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pycaw.pycaw import AudioUtilities

base_options = python.BaseOptions(model_asset_path = "hand_landmarker.task")
#configure hand landmarker, options for running mode are image, video and live stream, num_hands set to 1 for now

options = vision.HandLandmarkerOptions(
    base_options = base_options, 
    running_mode = vision.RunningMode.VIDEO, 
    num_hands = 1
)

#create landmarker for hands
landmarker = vision.HandLandmarker.create_from_options(options)

##create window
cv2.namedWindow("Hand tracking", cv2.WINDOW_NORMAL)
#open camera -- 0 is default
videoCap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
for _ in range(10):
    videoCap.read()

##get default speakers
device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume

#volume range
vol_range = volume.GetVolumeRange()
#read image
while True: 
    ##flip the image horizontally, convert the colors to RGB and wrap it for MediaPipe
    success, img = videoCap.read()
    if not success: 
        break
    img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #converts and saves color as rgb format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data= rgb)
    #timestamp to keep frame
    timestamp_ms = int(time.time() * 1000)

    ##frames for detecting vision (hand)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

      ##line connections for hand
    lines = [
                    #thumb
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    #index
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    #middle
                    (9, 10), (10, 11), (11, 12), 
                    #ring
                    (13, 14), (14, 15), (15, 16), 
                    #pinky
                    (17, 18), (18, 19), (19, 20), 
                    #palm
                    (5, 9), (9, 13), (13, 17), (0, 17)
                 ]

    ##make landmarks points
    if result.hand_landmarks: 
        hand = result.hand_landmarks[0]
        h, w, _ = img.shape

        points = []
        for landmark in hand: 
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append((x,y))

        ##thumb tip 
        x1, y1 = points[4]

        ##index tip
        x2, y2 = points[8]

        ##center point
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        ##calculate distance between fingers
        distance = math.hypot(x2 - x1, y2 - y1)

        ##volume percentage
        vol = np.interp(distance, [30, 220], [0,1])
        ##volume bar
        vol_bar = np.interp(distance, [30, 220], [0, 100])
        ##draw volume bar
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)

        cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)

        ##set system volume
        volume.SetMasterVolumeLevelScalar(vol, None)
        for start, end in lines:
          cv2.line(img, points[start], points[end], (0, 255, 0), 1)
        ##draw the point on hand - image, (x,y), radius of circle, color, thickness
        for point in points:
            cv2.circle(img, point, 5, (0,0, 255), -1)
    cv2.imshow("Hand tracking", img)

    ## to keep window running until q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
videoCap.release()
cv2.destroyAllWindows()