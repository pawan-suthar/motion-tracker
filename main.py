import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture('vidio/3.mp4')


ptime = 0
while True:
    success, img = cap.read()
    imS = cv2.resize(img, (800, 700))    
    
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
    results = pose.process(imgRGB) 
    print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(imS, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id , lm in enumerate(results.pose_landmarks.landmark):
            h ,w ,c = imS.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(imS,(cx,cy),5,(255,0,0), cv2.FILLED)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(imS, str(int(fps)),(70,50), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv2.imshow("image",imS)
    cv2.waitKey(10)