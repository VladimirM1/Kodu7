# Moodulite installeerimine ja importimine
import os

from ultralytics import YOLO
import cv2 as cv
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
model = YOLO("yolov9c.pt")

# Colabis vajalik teek piltide näitamiseks
#from google.colab.patches import cv2_imshow

# Moodulite importimine (klassifitseerimine ja tehisnärvivõrgud)
import tensorflow.keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, Activation, Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img


#Import vajalik selleks, et XML faili töödelda läbi
import xml.etree.ElementTree as ET

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import easyocr

#Proov

annotatsioonid = 'signs/annotations/'

def parse_annotatsioonid(xml):
  xml_parse = ET.parse(xml)
  root = xml_parse.getroot()

  kast = []
  kirjed = []
  yks_kirje = True
  for obj in root.findall('object'):
    name = obj.find('name').text
    kirjed.append(name)

    bndbox = obj.find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)
    kast.append((xmin,ymin,xmax,ymax))

  if (len(kast) != 1):
    yks_kirje = False

  return kast, kirjed, yks_kirje

images = []
kirjed = []

annotatsioonid_cnt = 0
pildid_cnt = 0

annotatsiooni_failid = os.listdir(annotatsioonid)
for annotatsioon in annotatsiooni_failid:
  if annotatsioon.endswith('.xml'):
    annotatsioonid_cnt += 1
    xml_teekond = os.path.join(annotatsioonid, annotatsioon)
    pildi_teekond = os.path.join('signs/images/', annotatsioon.replace('.xml','.png'))

    if os.path.exists(pildi_teekond):
      pildid_cnt += 1
      kast, kirje, yks_kirje = parse_annotatsioonid(xml_teekond)
      if yks_kirje:

        pilt = load_img(pildi_teekond)
        pilt_toarray = img_to_array(pilt)
        pilt = cv.resize(pilt_toarray, (64,64))

        images.append(pilt)
        kirjed.extend(kirje)

images = np.array(images, dtype='float32') / 255.0

kirjed = np.array(kirjed)
label_encoder = LabelEncoder()
kirjed = label_encoder.fit_transform(kirjed)
kirjed = to_categorical(kirjed)

#print(annotatsioonid_cnt)
#print(pildid_cnt)
print(images.shape)
print(kirjed.shape)

X_train, X_val, y_train, y_val = train_test_split(images,kirjed, test_size=0.2, random_state=42)

mudel = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])
mudel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = mudel.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

#loss, accuracy = mudel.evaluate(X_val, y_val)
#print(accuracy)

test_image = load_img('signs/test1.jpg')
plt.imshow(test_image)
test_image = img_to_array(test_image)
test_image = cv.resize(test_image,(64,64))
test_image = np.expand_dims(test_image, axis=0) / 255.0
predictions = mudel.predict(test_image)
predicted_kirje = label_encoder.inverse_transform([np.argmax(predictions)])
print(predicted_kirje)

test_image = load_img('signs/test2.jpg.webp')
plt.imshow(test_image)
test_image = img_to_array(test_image)
test_image = cv.resize(test_image,(64,64))
test_image = np.expand_dims(test_image, axis=0) / 255.0
predictions = mudel.predict(test_image)
predicted_kirje = label_encoder.inverse_transform([np.argmax(predictions)])
print(predicted_kirje)
"""
reader = easyocr.Reader(['en'])

def leia_kiirusepiirang(pilt):
  pilt_copy = pilt.copy()
  pilt = cv.resize(pilt, (64,64))
  pilt = np.expand_dims(pilt, axis=0) / 255.0

  predictions = mudel.predict(pilt)
  predicted_kirjed = label_encoder.inverse_transform([np.argmax(predictions)])

  if 'speedlimit' in predicted_kirjed:
    pilt_copy_conv = (pilt_copy * 255).astype(np.uint8)
    kiirusepiirang_vaartus = reader.readtext(pilt_copy_conv, detail=0)
    print(kiirusepiirang_vaartus)
    return kiirusepiirang_vaartus

  return None

test_kiiruse_piirang = load_img('signs/test70.jpg')
plt.imshow(test_kiiruse_piirang)
test_pilt_array = img_to_array(test_kiiruse_piirang)
kiiruse_piirang = leia_kiirusepiirang(test_pilt_array)
if kiiruse_piirang != None:
  print(kiiruse_piirang)
else:
  print("error")
"""


