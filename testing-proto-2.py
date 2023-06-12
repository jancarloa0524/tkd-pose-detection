import mediapipe as mp
import cv2
import os # work with filepaths
import numpy as np # array structures
from tensorflow.python.keras.models import load_model # loading model

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load model
model = load_model('action.h5')

# New Detection Variables

sequence = [] # collects our 30 frames, and passes into predictions model
sentence = [] # history of detections
word = ""
threshold = 0.9 # confidence metric. only render results if they are above this threshold

# make detections
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # color convert to RGB
    image.flags.writeable = False # sets image writable status to false
    results = model.process(image) # making predictions using mediapipe
    image.flags.writeable = True # sets image writable status to trye
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # color convert back BGR
    return image, results

# Techniques we detect
techniques = np.array(['complete snapkick!', 'pick up your knee first!','bring the knee back after kicking'])
# thirty videos worth of data
num_sequences = 90
# 30 frame length videos
sequence_length = 30
# storing each of our 30 frames (as numpy arrays) in different folders

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
            res = model.predict(np.expand_dims(sequence, axis=0))[0] # np.exapnd_dims the shape of X_test[0].shape is different from what the model expects. So use it to encapsulate it inside another array
        
            # Render predictions to screen
            if res[np.argmax(res)] > threshold: # if the accuracy score is above the threshold
                if len(sentence) > 0: # if sentence is greater than 0
                    if techniques[np.argmax(res)] != sentence[-1]: # check if the current action does not match the last prediction
                            sentence.append(techniques[np.argmax(res)])
                            word = techniques[np.argmax(res)]
                else:
                    sentence.append(techniques[np.argmax(res)])
            # if there are no sentences in the sentences array. If not, append it. If we do, check the current predicted word isn't the same as whatever is there. If it is, skip the append, but if not, append it. 
        
            # if the sentence is greater 2, then grab the last 2 values so we don't have a giant array
            if len(sentence) > 2:
                sentence = sentence[-2:]
        # render
        # cv2.rectangle(image, (0,0), (640,40), (245,117,16), -1)
        # cv2.putText(image, word, (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display Class, was 250
        cv2.rectangle(image, (0,0), (640, 30), (0,0,0), -1)
        cv2.putText(image, 'Action:'
                    , (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, word
                    , (100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        print(word)
        
        cv2.imshow('Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()