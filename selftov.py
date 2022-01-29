import cv2
import mediapipe as mp
import time
#import numpy as np
import math
#mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
imgs = []
cnt = 0

#normal:0
imgs.append(cv2.imread("./imgs/model-eye-n.png"))
#close:1
imgs.append(cv2.imread("./imgs/model-eye-c.png"))
#right-close:2
imgs.append(cv2.imread("./imgs/model-eye-r.png"))
#left-close:3
imgs.append(cv2.imread("./imgs/model-eye-l.png"))
#full:4
imgs.append(cv2.imread("./imgs/model-eye-f.png"))
#dark:5
imgs.append(cv2.imread("./imgs/model-dark.png"))
#smile:6
imgs.append(cv2.imread("./imgs/model-smile.png"))
#smile-c:7
imgs.append(cv2.imread("./imgs/model-smile-c.png"))
#smile-r:8
imgs.append(cv2.imread("./imgs/model-smile-r.png"))
#smile-l:9
imgs.append(cv2.imread("./imgs/model-smile-l.png"))
#surprise:10
imgs.append(cv2.imread("./imgs/model-surprise.png"))
#talk:11
imgs.append(cv2.imread("./imgs/model-talk.png"))
#talk2:12
imgs.append(cv2.imread("./imgs/model-talk2.png"))
#talk-c:13
imgs.append(cv2.imread("./imgs/model-talk-c.png"))



for i in range(0,len(imgs)):
    imgs[i] = cv2.resize(imgs[i],dsize=(900,654))



device = 0 # camera device number

def getFrameNumber(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * 1000 / fps)

    return frame_now
def distance(point1,point2):
    return math.sqrt(math.pow(point1[0]-point2[0],2)+math.pow(point1[1]-point2[1],2))


def drawFace(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    global cnt

    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue
        
        # Convert the obtained landmark values x and y to the coordinates on the image
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y, landmark_z])
    if len(landmark_point) != 0:

        #right-up 385~388
        cv2.circle(image, (int(landmark_point[385][0]),int(landmark_point[385][1])), 1, (0, 0, 255), 3) 
        #right-up-2 257~260
        cv2.circle(image, (int(landmark_point[257][0]),int(landmark_point[257][1])), 1, (0, 0, 255), 3) 
        #right-down 373~375
        #right-down-2 252~255
        cv2.circle(image, (int(landmark_point[374][0]),int(landmark_point[374][1])), 1, (0, 0, 255), 3)
        #mayu
        cv2.circle(image, (int(landmark_point[295][0]),int(landmark_point[295][1])), 1, (0, 0, 255), 3)
        #mejiri
        cv2.circle(image, (int(landmark_point[382][0]),int(landmark_point[382][1])), 1, (0, 0, 255), 3)
        cv2.circle(image, (int(landmark_point[263][0]),int(landmark_point[263][1])), 1, (0, 0, 255), 3)

        #left-up 158~160
        cv2.circle(image, (int(landmark_point[158][0]),int(landmark_point[158][1])), 1, (0, 0, 255), 3) 
        #left-up-2 27~29
        cv2.circle(image, (int(landmark_point[27][0]),int(landmark_point[27][1])), 1, (0, 0, 255), 3) 
        #left-down 145
        cv2.circle(image, (int(landmark_point[145][0]),int(landmark_point[145][1])), 1, (0, 0, 255), 3)
        #mayu
        cv2.circle(image, (int(landmark_point[65][0]),int(landmark_point[65][1])), 1, (0, 0, 255), 3)
        #mejiri
        cv2.circle(image, (int(landmark_point[130][0]),int(landmark_point[130][1])), 1, (0, 0, 255), 3)
        cv2.circle(image, (int(landmark_point[173][0]),int(landmark_point[173][1])), 1, (0, 0, 255), 3)
        
        #mouth
        cv2.circle(image, (int(landmark_point[11][0]),int(landmark_point[11][1])), 1, (255, 0, 0), 3)
        cv2.circle(image, (int(landmark_point[14][0]),int(landmark_point[14][1])), 1, (255, 0, 0), 3)
        cv2.circle(image, (int(landmark_point[291][0]),int(landmark_point[291][1])), 1, (255, 0, 0), 3)
        cv2.circle(image, (int(landmark_point[61][0]),int(landmark_point[61][1])), 1, (255, 0, 0), 3)

        #mouth
        mouth_width = distance(landmark_point[61],landmark_point[291])
        mouth_height = distance(landmark_point[11],landmark_point[14])
        mouth_params = mouth_height/mouth_width
        
        #right
        right_up_point = landmark_point[385]
        right_down_point = landmark_point[374]
        right_mayu_point = landmark_point[295]
        right_eye_width = distance(landmark_point[263],landmark_point[382])
        right_eye_height = distance(right_up_point,right_down_point)
        right_eye_height2 = distance(right_mayu_point,right_down_point)
        #left
        left_up_point = landmark_point[158]
        lef_mayu_point = landmark_point[65]
        left_down_point = landmark_point[145]
        left_eye_width = distance(landmark_point[173],landmark_point[130])
        left_eye_height = distance(left_up_point,left_down_point)
        left_eye_height2 = distance(lef_mayu_point,left_down_point)

        #smile
        smile_params=mouth_width/(left_eye_width+right_eye_width)

        #  --debag--
        #print((left_eye_height/left_eye_width),(right_eye_height/right_eye_width))
        #print(smile_params)
        #print(mouth_params)

        params = {"left":1,"right":1,"bigeyes":0,"mouth":0,"smile":0,"dark":0}

        if (left_eye_height/left_eye_width) < 0.28 and (right_eye_height/right_eye_width) < 0.32:
            params["left"] = 0
            params["right"] = 0
        elif (left_eye_height/left_eye_width) < 0.3 and (left_eye_height/left_eye_width)<(right_eye_height/right_eye_width):
            params["left"] = 0
        elif (right_eye_height/right_eye_width) < 0.31 and (left_eye_height2/left_eye_width)>(right_eye_height2/right_eye_width):
            params["right"] = 0
        elif (right_eye_height2/right_eye_width) > 1.1 and (left_eye_height2/left_eye_width) > 1.1 and cnt<=3:
            params["bigeyes"] = 1
            cnt += 0.2
        elif cnt>3 and (right_eye_height2/right_eye_width) > 1.1 and (left_eye_height2/left_eye_width) > 1.1:
            params["dark"] = 1
        else:
            cnt = 0
        if mouth_params > 0.2:
            params["mouth"] = 1
        if smile_params > 1.05:
            params["smile"] = 1
        
        if params["dark"] == 1:
            vimg = imgs[5]
        elif params["bigeyes"]==1 and params["mouth"]==1:
            vimg = imgs[10]
        elif params["bigeyes"]==1:
            vimg = imgs[4]
        elif params["smile"]==1 and params["left"]==0 and params["right"]==0:
            vimg = imgs[7]
        elif params["smile"]==1 and params["left"]==1 and params["right"]==0:
            vimg = imgs[8]
        elif params["smile"]==1 and params["left"]==0 and params["right"]==1:
            vimg = imgs[9]
        elif params["smile"]==1 and params["left"]==1 and params["right"]==1:
            vimg = imgs[6]
        elif params["mouth"]==1 and params["left"]==0 and params["right"]==0:
            vimg = imgs[13]
        elif params["mouth"]==1 and (params["left"]==0 or params["right"]==0):
            vimg = imgs[12]
        elif params["mouth"]==1 and params["left"]==1 and params["right"]==1:
            vimg = imgs[11]
        elif params["smile"]==0 and params["left"]==0 and params["right"]==0:
            vimg = imgs[1]
        elif params["smile"]==0 and params["left"]==1 and params["right"]==0:
            vimg = imgs[2]
        elif params["smile"]==0 and params["left"]==0 and params["right"]==1:
            vimg = imgs[3]
        else:
            vimg = imgs[0]

    return vimg
        
        

    

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

    #drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
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
                    vimg = drawFace(frame, face_landmarks)
            cv2.imshow('Myself', frame)
            cv2.imshow("vitural",vimg)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()

if __name__ == '__main__':
    main()
