from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import img_to_array
from keras.models import Sequential, Model, load_model
from matplotlib.pyplot import imshow
import numpy as np
import cv2
np.random.seed(42)
SIZE = 256

#Set to train
TRAIN = False

#Image processing
img_data1 = []
img_data2 = []
img1 = cv2.imread('league1.png')
img2 = cv2.imread('league2.png')
img1 = cv2.resize(img1, (SIZE, SIZE))
img2 = cv2.resize(img2, (SIZE, SIZE))
img_data1.append(img_to_array(img1))
img_data2.append(img_to_array(img2))

img_array1 = np.reshape(img_data1, (len(img_data1), SIZE, SIZE, 3))
img_array1 = img_array1.astype('float32')/255.
img_array2 = np.reshape(img_data2, (len(img_data1), SIZE, SIZE, 3))
img_array2 = img_array2.astype('float32')/255.

if TRAIN:    
    # Encoder part
    encoder = Sequential(name='encoder')
    #Expected half the size of previous layer
    #Layer1
    encoder.add(Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 3)))
    encoder.add(MaxPooling2D((2, 2), padding='same'))
    #Layer2
    encoder.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    encoder.add(MaxPooling2D((2, 2), padding='same'))
    #Layer3
    encoder.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    encoder.add(MaxPooling2D((2, 2), padding='same'))
    
    
    
    # Decoder part
    decoder = Sequential(name='decoder')
    #Expected twice the size of previous layer
    #Layer4
    decoder.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    decoder.add(UpSampling2D((2, 2)))
    #Layer5
    decoder.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    decoder.add(UpSampling2D((2, 2)))
    #Layer6
    decoder.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    decoder.add(UpSampling2D((2, 2)))
    
    #Activation layer
    decoder.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))  # Assuming 3 channels for RGB images
    
    autoencoder = Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    autoencoder.summary()
    
    autoencoder.fit(img_array1, img_array1, epochs=200, shuffle=True)
    prediction = autoencoder.predict(img_array1)
    
    reshaped_prediction_uint8 = (prediction[0].reshape(SIZE, SIZE, 3) * 255).astype(np.uint8)
    cv2.imshow('Prediction', reshaped_prediction_uint8)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
else:
    autoencoder = load_model('autoencoder_model.keras')
    prediction = autoencoder.predict(img_array1)
    prediction2 = autoencoder.predict(img_array2)
    
    imshow(prediction[0].reshape(SIZE, SIZE, 3))
    reshaped_prediction_uint8 = (prediction[0].reshape(SIZE, SIZE, 3) * 255).astype(np.uint8)
    imshow(prediction2[0].reshape(SIZE, SIZE, 3))
    reshaped_prediction2_uint8 = (prediction2[0].reshape(SIZE, SIZE, 3) * 255).astype(np.uint8)
    
    cv2.imshow('Prediction', reshaped_prediction_uint8)
    cv2.imshow('Prediction2', reshaped_prediction2_uint8)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

