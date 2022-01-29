import cv2
import mediapipe as mp
import time
import numpy as np
import pyautogui as pygui
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
#point[0]:x,  point[1]:y
point = [9,32]


device = 0 # camera device number

def getFrameNumber(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * 1000 / fps)

    return frame_now


def drawFace(image, landmarks):
    global point
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue
        
        # Convert the obtained landmark values x and y to the coordinates on the image
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y, landmark_z])
    x,y = pygui.position()
    x = x-point[0]
    y = y-point[1]
    if len(landmark_point) != 0:
        for i in range(0,len(landmark_point)):
            if x-2 <= landmark_point[i][0] <= x+2 and y-2 <= landmark_point[i][1]<= y+2:
                cv2.circle(image, (int(landmark_point[i][0]),int(landmark_point[i][1])), 1, (0, 255, 0), 5)
                print(i)
        
    print(x,y)

def main():
    # For webcam input:
    global device

    cap = cv2.VideoCapture(device)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print("Size:", ht, "x", wt, "/Fps: ", fps)

    start = time.perf_counter()
    frame_prv = -1
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            frame_now=getFrameNumber(start, fps)
            if frame_now == frame_prv:
                continue
            frame_prv = frame_now

            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            frame.flags.writeable = False
            results = face_mesh.process(frame)

            # Draw the face mesh annotations on the image.
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    drawFace(frame, face_landmarks)
            cv2.imshow('Search', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()

if __name__ == '__main__':
    main()
