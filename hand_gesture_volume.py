import cv2
import mediapipe as mp
import numpy as np
from math import hypot

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ================= AUDIO SETUP =================
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get volume range
volMin, volMax = volume.GetVolumeRange()[:2]

# ================= MEDIAPIPE =================
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Camera not working")
        break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    if len(lmList) != 0:
        # Thumb tip = 4, Index tip = 8
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        # Draw
        cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Distance
        length = hypot(x2 - x1, y2 - y1)

        # Convert distance → volume
        vol = np.interp(length, [30, 200], [volMin, volMax])
        volume.SetMasterVolumeLevel(vol, None)

        # Volume %
        volPer = np.interp(length, [30, 200], [0, 100])

        # Display
        cv2.putText(img, f'Volume: {int(volPer)} %', (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Hand Gesture Volume Control", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()