import cv2 as cv
import numpy as np
import joblib

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

import easyocr
import re

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

mudel = load_model('mudel.keras')
label_encoder = joblib.load('label_encoder.pkl')
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

    numeric_values = [re.findall(r'\d+', text) for text in kiirusepiirang_vaartus]
    numeric_values = [item for sublist in numeric_values for item in sublist]  # Flatten the list
    numeric_values = [int(value) for value in numeric_values]  # Convert to integers
    if numeric_values:
      #print(kiirusepiirang_vaartus)
      return numeric_values

  return None

def detect_roadsign(imagepath):
    test_image = load_img(imagepath)
    test_image = img_to_array(test_image)
    test_image = cv.resize(test_image, (64, 64))
    test_image = np.expand_dims(test_image, axis=0) / 255.0
    predictions = mudel.predict(test_image)
    predicted_kirje = label_encoder.inverse_transform([np.argmax(predictions)])
    if predicted_kirje == "speedlimit":
        test_kiiruse_piirang = load_img(imagepath)
        test_pilt_array = img_to_array(test_kiiruse_piirang)
        kiirusepiirangud = leia_kiirusepiirang(test_pilt_array)
        if kiirusepiirangud is None:
            return predicted_kirje[0].title() + " (Ei suutnud kiiruse numbrit lugeda)"
        if kiirusepiirangud[0] > 300:
            piirang = kiirusepiirangud[1]
        else:
            piirang = kiirusepiirangud[0]
        tagastatav = predicted_kirje[0].title() + " " + str(piirang)
        print(tagastatav)
        return tagastatav
    else:
        return predicted_kirje[0].title()

class RoadsignsDetector(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Roadsign detector")
        self.geometry("1000x600")

        self.upload_button = tk.Button(self, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=20)

        self.image_label = tk.Label(self)
        self.image_label.pack(pady=20)

        self.text_display = tk.Label(self, text="", wraplength=500)
        self.text_display.pack(pady=20)


    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])

        if file_path:
            image = Image.open(file_path)
            image.thumbnail((600, 400))
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            self.text_display.configure(text=detect_roadsign(file_path).title())

if __name__ == "__main__":
    app = RoadsignsDetector()
    app.mainloop()