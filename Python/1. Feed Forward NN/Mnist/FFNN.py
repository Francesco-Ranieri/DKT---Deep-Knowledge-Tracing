import tensorflow as tf 
import tensorflow.keras.layers as KL
import numpy as np 
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print (x_train.shape)

fig,ax = plt.subplots(10,10) 
k = 0
for i in range(10): 
    for j in range(10): 
        ax[i][j].imshow(x_train[k], aspect='auto') 
        k += 1
plt.show()

print (x_test.shape)

fig,ax = plt.subplots(10,10) 
k = 0
for i in range(10): 
    for j in range(10): 
        ax[i][j].imshow(x_test[k], aspect='auto') 
        k += 1
plt.show()

# create input layer
inputs = KL.Input(shape=(28, 28))                      #(None, 28, 28)

# flatten the inputs
l = KL.Flatten()(inputs)                               #(None, 784)

# print tensor shape
print ('input -->', inputs.shape)
print ('l -->', (KL.Flatten()(inputs)).shape)

l = KL.Dense(512, activation=tf.nn.relu)(l)            #(None, 512)

outputs = KL.Dense(10, activation=tf.nn.softmax)(l)    #(None, 10)

# print tensor shape
print (l.shape)
print (outputs.shape)

#create model (in, out)
model = tf.keras.models.Model(inputs, outputs)
model.summary()

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5)
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Loss: {0} - Test Acc: {1}".format(test_loss, test_acc))