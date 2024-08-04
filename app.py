from flask import Flask, render_template
app = Flask(__name__)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

mnist = tf.keras.datasets.mnist
(xtrain,ytrain),(xtest,ytest) = mnist.load_data()
xtrain.shape
ytrain.shape

xtrain[349]
ytrain[349]

# plt.imshow(xtrain[349],cmap='gray')
# plt.show()
# plt.figure(figsize=(15,15))
# for i in range(25):
#   plt.subplot(5,5,i+1)
#   plt.xticks([])
#   plt.yticks([])
#   plt.xlabel(ytrain[i],fontsize=15)
#   plt.imshow(xtrain[i],cmap='gray')
# plt.show()

# Create the model
model = tf.keras.models.Sequential()

# Add the layers in the model
# input layer
model.add(tf.keras.layers.Flatten()) ### input layer 28*28 matrix is flatten as 1*784 single layer
model.add(tf.keras.layers.Dense(784,activation='relu')) ### hidden layer
model.add(tf.keras.layers.Dense(10,activation='softmax')) ### output layer

# compile the model
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# scale the data
xtrain = xtrain/255  ## reduce the range from 0-255 to 0-1
xtest = xtest/255

xtrain[349]

# Train the model
model.fit(xtrain,ytrain,epochs=3)

predictions = model.predict(xtest)
ytest[23]
predictions[23]
np.argmax(predictions[23])
plt.figure(figsize=(15,15))
# for i in range(25):
#   plt.subplot(5,5,i+1)
#   plt.xticks([])
#   plt.yticks([])
#   plt.xlabel(np.argmax(predictions[i]),fontsize=15)
#   plt.imshow(xtest[i],cmap='gray')
# plt.show()

model.evaluate(xtest,ytest)

wrong = []
for i in range(0,len(xtest)):
  p = np.argmax(predictions[i])
  if p != ytest[i]:
    wrong.append(i)

len(wrong)

# for i in wrong:
#   plt.xticks([])
#   plt.yticks([])
#   plt.xlabel(np.argmax(predictions[i]),fontsize=15)
#   plt.imshow(xtest[i],cmap='gray')
#   plt.show()

#### test with the real image
img_array = cv2.imread("iiim.png", 0)  ### read the image and convert to array
file1="iiim.png"

plt.imshow(img_array,cmap='gray')
# plt.show()

new_array = cv2.bitwise_not(img_array)
plt.imshow(new_array,cmap='gray')
# plt.show()

new_array.shape
new_array = cv2.resize(new_array,(28,28))
plt.imshow(new_array,cmap='gray')
# plt.show()

### scale the image
new_array = new_array/255
plt.imshow(new_array,cmap='gray')
# plt.show()

new_array.shape
predicted_number = model.predict(np.array([[new_array]]))

result=np.argmax(predicted_number)

@app.route('/')
def hello_world():
    return render_template('./index.html',result=result,file1=file1)