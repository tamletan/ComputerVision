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

thread_hold = 0.6
dataPath = 'data'
testPath ='test/test-images'
dir_name = []

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
    os.chdir(dataPath)
    dir_name = [n for n in os.listdir() if os.path.isdir(n) and not n.startswith('.')]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    #Feature extraction:
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40,
        thresholds=[0.6, 0.7, 0.7], factor=0.709)
    #Crop face from image:
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    os.chdir('..')

    model = Sequential([
        Dense(64, activation='relu', input_shape=(512,)),
        Dense(64, activation='relu'),
        Dense(len(dir_name), activation='softmax'),
    ])

    # Load the model's saved weights.
    model.load_weights('model.h5')
    return mtcnn,resnet,device,model

def pred_image(mtcnn, resnet, device, model):
    list_test_folder = [testPath+"/"+n for n in os.listdir(testPath) if not n.startswith('.')]
    list_test_images = []
    for i in list_test_folder:
    	images_in_folder = [i+"/"+n for n in os.listdir(i) if not n.startswith('.')]
    	list_test_images.extend(images_in_folder)
    test_images =[]

    for image in list_test_images:   
        im = cv2.imread(image)
        if isinstance(im, type(None)):
            print('[Error]: ',image)
            continue
        im_crop = mtcnn(im)
        if isinstance(im_crop, type(None)):
            print('[Error]: ',image)
            continue
        im_crop = im_crop.to(device)
        im_extract = resnet(im_crop.unsqueeze(0))
        im_extract = im_extract.detach().cpu().numpy()
        im_extract = im_extract.reshape(512)
        test_images.append(im_extract)

    test_images = np.asarray(test_images)

    #predict the images:
    predictions = model.predict(test_images[:])
    resProb = np.amax(predictions, axis=1)
    res = np.argmax(predictions, axis=1)
    corr_count = 0

    for i in range(len(res)):
        if(resProb[i] >= thread_hold):
            print(list_test_images[i], ":", "res: ", dir_name[res[i]], ":", "prob: ", resProb[i])
            if list_test_images[i].split('/')[-2] == dir_name[res[i]]:
                corr_count += 1
        else:
            print(list_test_images[i], ":", "Unknown")

    print('ListImageFolder: ', len(list_test_images))
    print('Correct Acc: {:.0%}'.format(corr_count/len(list_test_images)))

if __name__ == '__main__':
	np.set_printoptions(precision=3, suppress=True)
	mtcnn, resnet, device, model = sys_init()
	pred_image(mtcnn,resnet,device, model)