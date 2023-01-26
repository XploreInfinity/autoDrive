# train_model.py
LR = 0.01
WIDTH = 270
HEIGHT = 480
import numpy as np
from irv2_04 import model
EPOCHS = 10
MODEL_NAME = 'car-fast-{}-{}-{}-epochs-300K-data.model'.format(LR, 'irv2_04',EPOCHS)

hm_data = 22
for i in range(EPOCHS):
    for i in range(1,hm_data+1):
        # train_data = np.load('training_data-{}-balanced.npy'.format(i), allow_pickle=True)
        train_data = np.load('training_data.npy', allow_pickle=True)

        train = train_data[:-100]
        test = train_data[-100:]

        X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,3)
        Y = [i[1] for i in train]

        test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,3)
        test_y = [i[1] for i in test]

        model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}), 
            snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

        model.save(MODEL_NAME)



# tensorboard --logdir=foo:C:/path/to/log





