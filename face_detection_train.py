from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import multiprocessing as mp
import os
import cv2
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

dataPath = 'data'  #create dataset directory
dir_name =[]

def sys_init():
	# set gpu memory growth True
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
		try:
			for gpu in gpus:
				tf.config.experimental.set_memory_growth(gpu, True)
		except RuntimeError as e:
			print(e)

	global dir_name
	global dataPath
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	os.chdir(dataPath)
	dir_name = [n for n in os.listdir() if os.path.isdir(n) and not n.startswith('.')]
	print('Running on device: {}'.format(device))

	#Feature extraction:
	mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40,
	    thresholds=[0.6, 0.7, 0.7], factor=0.709)
	#Crop face from image:
	resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

	#create model
	model = Sequential([
	Dense(64, activation='relu', input_shape=(512,)),
	Dense(64, activation='relu'),
	Dense(len(dir_name), activation='softmax'),])

	return mtcnn,resnet,device, model

def training_model(mtcnn, resnet, device, model):
	global dir_name
	print(dir_name)
	train_images = []
	train_labels = []

	count=0
	for person in dir_name:
		images_list = [i for i in os.listdir(person) if not i.startswith('.')]

		for path in images_list:
			path = person+"/"+path
			im = cv2.imread(path)
			if isinstance(im, type(None)):
				print('[Error]: ',path)
				continue
			im_crop = mtcnn(im)
			if isinstance(im_crop, type(None)):
				print('[Error]: ',path)
				continue
			im_crop = im_crop.to(device)
			im_extract = resnet(im_crop.unsqueeze(0))
			im_extract = im_extract.detach().cpu().numpy()
			im_extract = im_extract.reshape(512)
			train_images.append(im_extract)
			train_labels.append(count)
			print("Done!")
		count+=1
	train_images = np.asarray(train_images)
	print('')

	model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
	)

	model.fit(
	    train_images,
	    to_categorical(train_labels),
	    epochs=8,
	    batch_size=32,
	)
	os.chdir('..')
	model.save_weights('model.h5')    

if __name__ == '__main__':
	mtcnn, resnet, device, model = sys_init()
	training_model(mtcnn,resnet,device, model)