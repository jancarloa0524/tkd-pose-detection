import mediapipe as mp
import cv2
import os # work with filepaths
import numpy as np # array structures
from playsound import playsound


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# make detections
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # color convert to RGB
    image.flags.writeable = False # sets image writable status to false
    results = model.process(image) # making predictions using mediapipe
    image.flags.writeable = True # sets image writable status to trye
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # color convert back BGR
    return image, results

# We create the file path for the project data here
# path for exported numpy arrays
DATA_PATH = os.path.join('Data')
# techniques we detect
# , 'pick up your knee first!','bring the knee back after kicking'
techniques = np.array(['bring the knee back after kicking'])
# thirty videos worth of data
num_sequences = 90
# 30 frame length videos
sequence_length = 30
# storing each of our 30 frames (as numpy arrays) in different folders
for technique in techniques:
	for sequence in range(num_sequences):
		try:
			os.makedirs(os.path.join(DATA_PATH, technique, str(sequence)))
			# create a folder called MP_Data, then a subfolder for the action, then a subfolder for the sequence
		except:
			pass

cap = cv2.VideoCapture(0)

# Initiate holistic model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        for technique in techniques:
            for sequence in range(num_sequences):
                    for frame_num in range(sequence_length):
                        ret, frame = cap.read()
                        
                        # make detections
                        image, results = mediapipe_detection(frame, pose)
                        
                        # pose detections
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                        if frame_num == 0:
                            if(sequence == 0 and technique != techniques[0]):
                                  playsound('change_action.mp3')
                            cv2.rectangle(image, (0,0), (800, 20), (0,0,0), -1)
                            cv2.rectangle(image, (150,265), (515,315), (0,0,0), -1)
                            cv2.putText(image, 'STARTING COLLECTION', (160,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 4, cv2.LINE_AA)
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(technique, sequence), (150,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                            # show to screen
                            cv2.imshow('OpenCV Feed', image)
                            playsound('end_sequence.mp3')
                            cv2.waitKey(1000) # 2 second break between each video
                            playsound('change_sequence.mp3')
                        else: # since we aren't starting collection anymore, we can just display the video number
                            cv2.rectangle(image, (550,0), (800, 40), (0,0,0), -1)
                            cv2.putText(image, str(frame_num), (580, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                            # show to screen
                            cv2.imshow('OpenCV Feed', image)
                        
                        # save each frame as a np array, resulting in 30 np arrays for each sequence
                        pose_keypoints = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
                        npy_path = os.path.join(DATA_PATH, technique, str(sequence), str(frame_num)) # where we are saving our frame
                        np.save(npy_path, pose_keypoints) # save the frame

                        if frame_num == 29 and sequence == 29 and technique == techniques[techniques.size - 1]:
                            playsound('end_sequence.mp3')

                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
        break
    
cap.release()
cv2.destroyAllWindows()