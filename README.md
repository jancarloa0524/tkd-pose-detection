# tkd-pose-detection

To access logs, `tensorboard --logdir logs/fit` in cmd line

Ensure the library `playsound` is version 1.2.2
`pip install playsound==1.2.2`

## Ensure the following values are the same across all files:
``` python 
DATA_PATH = os.path.join('datapath')
# Actions we detect
actions = np.array(['array of data values'])
# thirty videos worth of data
num_sequences = 30
# 30 frame length videos
sequence_length = 30
```

Make certain that in `training-proto-2.py`, that `input_shape=(x, 132)` where `x = sequence_length`

Make certain that in `testing-proto-2.py` that `sequence = sequence[-(x):]` and ` if len(sequence) == 30:`
where `x = sequence_length`
