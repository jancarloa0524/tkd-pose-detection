import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils # mediapipe drawing
mp_pose = mp.solutions.pose # mediapipe pose solution

# Values for circle position
# Array of coords
coords = np.array([[[530, 240], [510,150], [490,60], [510,330], [490,420]], [[110, 240], [130, 150], [150, 60], [130, 330], [150, 420]]])
selectCoords = 0

averageXPos = 0
side = 0
chosenLandmark = 31;

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # color convert to RGB
    image.flags.writeable = False # sets image writable status to false
    results = model.process(image) # making predictions using mediapipe
    image.flags.writeable = True # sets image writable status to trye
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # color convert back BGR
    return image, results

cap = cv2.VideoCapture(0)

# Initiate holistic model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read() 
        
        # make detections
        image, results = mediapipe_detection(frame, pose)
        
        # render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # collect keypoints
        pose_keypoints = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros(132) 
        # Based on where the user is, place the circle and select the landmark the user needs to use
        try:
            # average the X values, while removing the chosen landmark so that it does not interfere with where the user is
            averageXPos = np.mean(pose_keypoints, where=[[True, False, False, False]])
            averageXPos = ((averageXPos * 33) - pose_keypoints[chosenLandmark][0]) / 32
            # if the user is on the left side, the circle should be on the right and the user needs to use the right side of their body, and vice versa
            if averageXPos < 0.5:
                side=0 #right
                chosenLandmark = 31
            else:
                side=1 #left
                chosenLandmark = 32
        except:
            pass

        # Display 
        try:

            cv2.ellipse(image, coords[side][selectCoords], (30,30), 0, 0, 360, (0,0,255), -1)
            
            # pose_keypoints[landmark][x,,y, or z] * 640 or *480 to match screen res. coords[either right or left circles][selectCoords: a randomly chosen coordinate]
            if ((pose_keypoints[chosenLandmark][0] * 640) > coords[side][selectCoords][0] - 15 and (pose_keypoints[chosenLandmark][0] * 640) < coords[side][selectCoords][0] + 15 and (pose_keypoints[chosenLandmark][1] * 480) > coords[side][selectCoords][1] - 15 and (pose_keypoints[chosenLandmark][1] * 480) < coords[side][selectCoords][1] + 15):
                selectCoords = np.random.choice(5)
            cv2.imshow('Feed', image)
        except:
            pass
            cv2.putText(image, 'MUST BE IN FRAME', (160,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 4, cv2.LINE_AA)
            cv2.imshow('Feed', image)


        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()