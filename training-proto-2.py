from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import os #working with filepaths
import numpy as np #structuring datasets
import datetime # tracking logs
import tensorflow as tf # logging model performance with Tensorboard
from tensorflow.python.keras.models import Sequential # Sequential Nueralnetwork
from tensorflow.python.keras.layers import LSTM, Dense # LSTM layer of NueralNetwork, allows us to perform action-deteciton. Dense is a normal, fully connected layer 

# path for numpy arrays
DATA_PATH = os.path.join('MP_Data')
# Actions we detect
actions = np.array(['jab', 'cross','hook'])
# thirty videos worth of data
num_sequences = 30
# 30 frame length videos
sequence_length = 30

label_map = {label:num for num, label in enumerate(actions)}

# sequences represents x data, labels are y data
sequences, labels = [], []
# loop through each action and sequence
for action in actions:
    for sequence in range(num_sequences):
        window = [] # represents all frames for that particular sequence
        for frame_num in range(sequence_length): # loop through each frame
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))) # load the np array
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

#np.array(sequences).shape
# 90 videos, 30 frames, 132 keypoints
# pre-process data for LSTM neural network
X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05) # partition data for train/test split
# Build and Train LSTM Network

# initilialize sequential model
model = Sequential()
# Add 3 sets of LSTM layers, and 3 sets of Dense layers
model.add(LSTM(64, return_sequences = True, activation='relu', input_shape=(30,132))) # must have return sequences unitl the last one; each video is 30 frames by 132 keypoints
model.add(LSTM(128, return_sequences = True, activation='relu'))
model.add(LSTM(64, return_sequences = False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax')) # the return output will be the action that is done

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy']) # since we are doing a multi-class classification model, we need categorical tools

# Logging to tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(X_train, y_train, epochs=60, callbacks=[tensorboard_callback]) # train our model

model.save('action.h5')
