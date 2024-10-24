import cv2 
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical 


image_directory ='data set/'

no_tumor_images=os.listdir(image_directory+ 'no/')
yes_tumor_images=os.listdir(image_directory+ 'yes/') # gets all the images in the folder, that have no tumor
dataset=[]
label=[]

INPUT_SIZE =64
#print(no_tumor_images)

#path ='no0.jpg'

#print(path.split('.')[1])

for i, image_name in enumerate(no_tumor_images):    #allows to loop over while keeping track of both the index and the value of each item
    if(image_name.split('.')[1]=='jpg'):   #make sure the photo is in JPEG
        image=cv2.imread(image_directory+'no/'+image_name) #loads photos, the one with no tumors, and creates file path for each image, and stores it in image
        image= Image.fromarray(image, 'RGB') #will convert the image to RGB format
        image=image.resize((INPUT_SIZE,INPUT_SIZE)) #resizes the image to 64x64 pixels
        dataset.append(np.array(image))#coverts the image to a numpy array, and appends it to the dataset list
        label.append(0) #adds a label of 0 to the label list, since the image is of a no tumor


#this is the same as the above, but for the images with tumors
for i, image_name in enumerate(yes_tumor_images):    #allows to loop over while keeping track of both the index and the value of each item
    if(image_name.split('.')[1]=='jpg'):   #make sure the photo is in JPEG
        image=cv2.imread(image_directory+'yes/'+image_name) #loads photos, the one with tumors, and creates file path for each image, and stores it in image
        image=Image.fromarray(image, 'RGB') #will convert the image to RGB format
        image=image.resize((INPUT_SIZE,INPUT_SIZE)) #resizes the image to 64x64 pixels
        dataset.append(np.array(image))#coverts the image to a numpy array, and appends it to the dataset list
        label.append(1) #adds a label of 1 to the label list, since the image is of a tumor

dataset=np.array(dataset) #coverts the dataset to a numpy array, to better use as numerical data
label= np.array(label)

#using 20 percent of the data for testing,
x_train, x_test, y_train, y_test= train_test_split(dataset, label, test_size=0.2, random_state=0)

#reshape = (n, image_width, image_height, n_channel) 

#print(x_train.shape) # result (2400 (80 percent for training), 64,64 resizing, and 3 is RGB (red,green, blue)
#print(y_train.shape)# result (2400 (80 percent for training), 64,64 resizing, and 3 is RGB (red,green, blue)


#print(x_test.shape)# result (600 (20 percent for testing), 64,64 resizing, and 3 is RGB (red,green, blue)
#print(y_test.shape)# result (600 (20 percent for testing), 64,64 resizing, and 3 is RGB (red,green, blue)

#normalize data
x_train= normalize(x_train, axis=1) #axis=1 means normalize along the columns
x_test= normalize(x_test, axis=1)

y_train= to_categorical(y_train, num_classes=2) #converts the labels to a categorical format
y_test= to_categorical(y_test, num_classes=2)

#build the model
model=Sequential() #creates a sequential model

#adds a convolutional layer with 32 filters, each filter is 3x3, and the input shape is 64x64x3
model.add((Conv2D(32, (3,3), input_shape=(INPUT_SIZE,INPUT_SIZE,3))))
model.add(Activation('relu')) #adds a activation layer
model.add(MaxPooling2D(pool_size=(2,2))) #adds a max pooling layer

#adds a convolutional layer with 32 filters, each filter is 3x3, and the input shape is 64x64x3
model.add((Conv2D(32, (3,3), kernel_initializer='he_uniform'))) #
model.add(Activation('relu')) #adds a activation layer
model.add(MaxPooling2D(pool_size=(2,2))) #adds a max pooling layer

#adds a convolutional layer with 32 filters, each filter is 3x3, and the input shape is 64x64x3
model.add((Conv2D(64, (3,3), kernel_initializer='he_uniform')))
model.add(Activation('relu')) #adds a activation layer
model.add(MaxPooling2D(pool_size=(2,2))) #adds a max pooling layer

model.add(Flatten()) #adds a flatten layer
model.add(Dense(64)) #adds a dense layer
model.add(Activation('relu')) #adds a activation layer (simple cnn model)
model.add(Dropout(0.5)) #adds a dropout layer
model.add(Dense(2)) #adds a dense layer (it is one because we are using binary classification problem) - binary cross entropy 
model.add(Activation('softmax')) #adds a activation layer

#binary CrossEntropy= 1
#Categorical Cross Entrophy = 2 , softmax 

#compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #compiles the model

#train the model
model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10, validation_data=(x_test, y_test), shuffle=False)
#verbose=1 means it will print out the progress of the training
#epochs=10 means it will train the model 10 times
#validation_data=(x_test, y_test) means it will use the test data to evaluate the model

model.save('BrainTumor10EpochsCategorical.h5') #saves the model






       
