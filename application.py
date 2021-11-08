# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
from keras.applications.xception import Xception, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input

# other imports
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import cv2
import h5py
import os
import json
import pickle as cPickle
import datetime
import time
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

lab = ["Pepper_bell_Bacterial_spot","Pepper__bell___healthy","Potato___Early_blight","Potato___healthy","Potato___Late_blight","Tomato__Target_Spot","Tomato__Tomato_mosaic_virus","Tomato__Tomato_YellowLeaf__Curl_Virus","Tomato_Bacterial_spot","Tomato_Early_blight","Tomato_healthy","Tomato_Late_blight","Tomato_Leaf_Mold","Tomato_Septoria_leaf_spot","Tomato_Spider_mites_Two_spotted_spider_mite"]

# load the user configs
with open('conf/conf.json') as f:    
	config = json.load(f)

# config variables
model_name = config["model"]
weights = config["weights"]
include_top = config["include_top"]
train_path = config["train_path"]
features_path = config["features_path"]
labels_path = config["labels_path"]
#test_size = config["test_size"]
results = config["results"]
model_path = config["model_path"]
classifier_path = config["classifier_path"]
# create the pretrained models
# check for pretrained weight usage or not
# check for top layers to be included or not
base_model = Xception(weights=weights)
model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
image_size = (299, 299)

# base_model = InceptionResNetV2(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
# model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
# image_size = (299, 299)

# base_model = MobileNet(include_top=include_top, weights=weights, input_tensor=Input(shape=(224,224,3)), input_shape=(224,224,3))
# model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
# image_size = (224, 224)

print ("[INFO] successfully loaded base model and model...")

loaded_model = cPickle.load(open(classifier_path, 'rb'))

print ("[INFO] successfully Loaded Trained Model...")

cur_path = "test"
for test_path in glob.glob(cur_path + "/*.jpg"):
	#load = i + ".png"
	print ("[INFO] loading", test_path,"image ")
	img = image.load_img(test_path, target_size=image_size)
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	feature = model.predict(x)
	#flat = feature.flatten()
	preds = loaded_model.predict(feature)
	print (preds)
	print ("I think the disease is : ",lab[preds[0]])
	show_image = cv2.imread(test_path)
	show_image = cv2.resize(show_image, (500, 500)) 
	#disease = preds
	#print (show_image)
	cv2.putText(show_image, lab[preds[0]], (40,50), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 0), 2)
	cv2.imshow("result",show_image)
	cv2.waitKey(0)