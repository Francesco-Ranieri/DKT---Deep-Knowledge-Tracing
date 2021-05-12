import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)

"""We need to load the IMDB dataset. We are constraining the dataset to the top 5,500 words. We also split the dataset into train (50%) and test (50%) sets"""

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)

print(f'x_train shape: {x_train.shape}')
print(f'y_train shape: {y_train.shape}')

"""Next, we need to truncate and pad the input sequences so that they are all the same length for modeling. The model will learn the zero values carry no information so indeed the sequences are not the same length in terms of content, but same length vectors is required to perform the computation in Keras."""

# truncate and pad input sequences
max_review_length = 500
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

"""**First layer** is the Embedded layer that uses 32 length vectors to represent each word

**Second layer** is the LSTM layer with 100 memory units

**Dense output layer** with a single neuron and a sigmoid activation function to make 0 or 1 predictions for the two classes (good and bad) in the problem.
"""

# create the model
embedding_vecor_length = 32
model = Sequential()

#input layer
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))

#second layer
model.add(LSTM(100))

#output layer
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=64)

scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))