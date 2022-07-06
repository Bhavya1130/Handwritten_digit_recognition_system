# import the libraries
import tensorflow as tf
from tensorflow import keras
from keras import layers
import  numpy as np


from keras.layers import Flatten, Dense
from keras import Sequential

# Object of the MNIST dataset
mnist = keras.datasets.mnist

# Load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Import Matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

# Normalize the train and test dataset
x_train = x_train/255.0
x_test = x_test/255.0

# Train the Model
model = tf.keras.models.Sequential([
    Flatten(input_shape = (28, 28)),
    Dense(128, activation = "relu"),
    Dense(10, activation = "softmax")
  ])

# Compile the Model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics = ["accuracy"])

# Print the summary of the Model
model.summary()

# Train the model
model.fit(x_train, y_train, epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)

# Print out the model accuracy and loss
print('test accuracy: ', test_acc)
print('test loss: ', test_loss)

# Make prediction
predictions = model.predict([x_test]) 

# Print out the number
print('Enter a number(as index) between 0 to 9999 \n\nYou will get the number at that index as output\n')
n = int(input())
if n>0 & n<10000 :
  print("\nThe number at index ", format(n), " is: ")
  print(np.argmax(predictions[n])) 
else:
  print('Enter number in the range')
  
  # Plot out the number
print('Here is the image of the number at index: ', format(n))
if n>0 & n<10000 :
  plt.imshow(x_test[n], cmap="gray") # Import the image
  plt.show() # Show the image
else:
  print('Enter number in the range')
