
from models import inception_v3 as googlenet
from alexnet import alexnet



def loader1():
	WIDTH = 60
	HEIGHT = 80
	LR = 1e-3
	EPOCHS = 10
	MODEL_NAME = 'python-drives-{}-{}-{}-epochs.model'.format(LR,'alexnetv2',EPOCHS)
	model = googlenet(WIDTH, HEIGHT, 3, LR, output=3, model_name=MODEL_NAME)
	model.load('/home/nayangupta824/python_drives/Final_training_data/'+MODEL_NAME)
	return model
	
def loader2():
	WIDTH = 60
	HEIGHT = 80
	LR = 1e-3
	EPOCHS = 8
	MODEL_NAME = 'python-drives-{}-{}-{}-epochs.model'.format(LR,'alexnetv2',EPOCHS)
	model = alexnet(WIDTH, HEIGHT, LR)
	model.load('/home/nayangupta824/python_drives/model_old1/'+MODEL_NAME)
	return model
