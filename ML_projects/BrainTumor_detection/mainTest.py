import cv2 #to import the images
from keras.models import load_model #to load the model
from PIL import Image #to load the image
import numpy as np #to convert the image to a numpy array

model=load_model('BrainTumor10Epochs.h5') #loads the model

image= cv2.imread('C:\\Users\\roger\Downloads\\archive\\pred\\pred0.jpg') #loads the image

img=Image.fromarray(image) #converts the image to a PIL image

img= img.resize((64,64)) #resizes the image

#convert image to numpy array

img=np.array(img)
 #converts the image to a numpy array

input_img= np.expand_dims(img, axis=0) #adds an extra dimension to the image

result= model.predict(input_img) #predicts the image
predicted_class= int(np.argmax(result, axis=1)) #gets the predicted class
print(result) #prints the result


