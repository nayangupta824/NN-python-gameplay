import numpy as np
from models import inception_v3 as googlenet

WIDTH = 60
HEIGHT = 80
#learning rate
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'python-drives-{}-{}-{}-epochs.model'.format(LR,'alexnetv2',EPOCHS)



model = googlenet(WIDTH, HEIGHT, 3, LR, output=3, model_name=MODEL_NAME)

train_data = np.load('/home/nayangupta824/python_drives/Final_training_data/training_data_final.npy')

#split to train and test data
train = train_data[:-100]
test = train_data[-100:]

X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
Y = [i[1] for i in train]


test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
test_y = [i[1] for i in test]



model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), 
snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save('/home/nayangupta824/python_drives/Final_training_data/'+MODEL_NAME)


