import numpy as np
import pandas as pd
import multiprocessing as mp
import os
import cv2
import torch
import tensorflow as tf
import time
import pickle

from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array
import pandas as pd

thread_hold = 0.5
dataPath = 'data'
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
	mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40, post_process=False,
		thresholds=[0.6, 0.7, 0.7], factor=0.709, keep_all=True, device=device)

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
	live_model = load_model('liveness/liveness.model')
	le = pickle.loads(open('liveness/le.pickle', 'rb').read())

	return mtcnn,resnet,device,model,live_model,le

def recognize_faces(mtcnn, resnet, device, model, live_model, le) :
	cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
	font = cv2.FONT_HERSHEY_SIMPLEX

	start_time = time.time()

	while True :
		_, image = cap.read()
		if cv2.waitKey(1) & 0xFF == ord('q') : # exit on q
			break

		boxes, _ = mtcnn.detect(image)

		if not isinstance(boxes, type(None)):
			for (x, y, w, h) in boxes:
				x = int(x)
				y = int(y)
				w = int(w)
				h = int(h)

				im_crop = image[y:h, x:w]
				try:
					im_crop = cv2.resize(im_crop, (160, 160), interpolation = cv2.INTER_AREA)
				except Exception as e:
					print(str(e))
					continue
				if not pred_liveness(im_crop, live_model, le):
					cv2.rectangle(image, (x, y), (w,h), (255, 0, 0), 2)
					cv2.putText(image, "Spoofing", (x+5,y-5), font, 1, (255,0,0), 2)
					continue
				cv2.rectangle(image, (x, y), (w,h), (0, 255, 0), 2)
				im_crop = ToTensor()(im_crop)
				im_crop = im_crop.to(device)

				im_extract = resnet(im_crop.unsqueeze(0))

				im_extract = im_extract.detach().cpu().numpy()
				im_extract = im_extract.reshape(512)

				predictions = model.predict(np.asarray([im_extract]))
				resProb = np.amax(predictions)
				res = np.argmax(predictions)

				if(resProb >= thread_hold):
					cv2.putText(image, '{0}: {1:.1%}'.format(dir_name[res], resProb), (x+5,y-5), font, 1, (0,0,255), 2)
					attendance(dir_name[res])
				else:
					cv2.putText(image, "Unknown", (x+5,y-5), font, 1, (255,0,0), 2)

			cv2.imshow("Face Recognizer", image)
		else:
			cv2.imshow("Face Recognizer", image)
			start_time = time.time()
			continue

		print("\rFPS: {:.2f}".format(1 / (time.time() - start_time)), end='')
		start_time = time.time()

	cap.release()
	cv2.destroyAllWindows()

def pred_liveness(crop, live_model, le):
	face = crop.copy()
	try :
		face = cv2.resize(face, (32, 32))
		face = face.astype("float") / 255.0
		face = img_to_array(face)
		face = np.expand_dims(face, axis=0)
	except Exception as e:
		print(str(e))

	preds = live_model.predict(face)[0]
	j = np.argmax(preds)
	if preds[j] < thread_hold:
		return False
	label = le.classes_[j]
	if label=='real':
		return True
	else:
		return False

def load_students_list():
	pd.options.mode.chained_assignment = None
	students_df = pd.read_csv('./students.csv')
	return students_df

def attendance(str_name):
	df.at[ df[df['Name'] == str_name].index , 'Attendance' ] = 1
	df.to_csv('./students.csv', index=False)

if __name__ == '__main__':
	df = load_students_list()
	torch.cuda.init()
	mtcnn, resnet, device, model, live_model, le = sys_init()
	torch.cuda.set_device(device)
	recognize_faces(mtcnn,resnet,device, model, live_model, le)