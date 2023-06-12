# Basically, instead of
import mediapipe as mp
import cv2
import os # work with filepaths
import numpy as np # array structures
from tensorflow.python.keras.models import load_model # loading model
from playsound import playsound

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load model
model = load_model('iteration_2_FOR_EXPO/action.h5')

# We create the file path for the project data here
# techniques we detect
techniques = np.array(['jab', 'cross','snapkick'])
# thirty videos worth of data
num_sequences = 30
# 30 frame length videos
sequence_length = 30
# storing each of our 30 frames (as numpy arrays) in different folders

# New Detection Variables

sequence = [] # collects our 30 frames, and passes into predictions model
sentence = [] # history of detections
requiredTech = np.random.choice(techniques)
B = 255
R = 255
threshold = 0.9 # confidence metric. only render results if they are above this threshold

# make detections
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

        # Prediction Logic
        pose_keypoints = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
        sequence.append(pose_keypoints)# inserting keypoints in sequence
        sequence = sequence[-30:] # grab the last 30 frames

        if len(sequence) == 30: # only run prediction if there are 30 frames
            if (B == 0):
                requiredTech = np.random.choice(techniques)
            B = 255
            R = 255
            res = model.predict(np.expand_dims(sequence, axis=0))[0] # np.expand_dims the shape of X_test[0].shape is different from what the model expects. So use it to encapsulate it inside another array
            # Render predictions to screen
            if res[np.argmax(res)] > threshold: # if the accuracy score is above the threshold
                if requiredTech == techniques[np.argmax(res)]:
                    B = 0
                    R = 0
                    playsound('change_sequence.mp3')
                    sequence = []
            # if there are no sentences in the sentences array. If not, append it. If we do, check the current predicted word isn't the same as whatever is there. If it is, skip the append, but if not, append it. 
            

        # Display Class
        cv2.rectangle(image, (0,0), (350, 40), (0,0,0), -1)
        cv2.putText(image, 'Required Action:'
                    , (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, requiredTech
                    , (225, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (B, 255, R), 1, cv2.LINE_AA)
        
        cv2.imshow('Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()