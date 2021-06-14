import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers, optimizers, losses, metrics
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import keras
from keras.layers.embeddings import Embedding
from keras import backend as K
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
import cv2
import os
import glob
import matplotlib.pyplot as plt
import timeit
from sklearn.metrics import classification_report, confusion_matrix
os.environ['KMP_DUPLICATE_LIB_OK']='True'


start_time = timeit.default_timer()

# ------------ pre-processing data --------------

labels = {
    "grain" : 0,
    "earn" : 1,
    "acq" : 2,
    "crude" : 3,
    "money-fx" : 4,
    "interest" : 5
}

class Vocabulary:
    def __init__(self):
        self.no_word = 1
        self.wordIndexDic = {"UNK":0} # stores word and corresponding index
    
    def addToDicionary(self, word):
        # if word exists in dictionary add it
        if word not in self.wordIndexDic:
            self.wordIndexDic[word] = self.no_word
            self.no_word += 1

    def getIndex(self, word):
        # if its in the dictionary then return it's ID, otherwise its an unknown word
        if word in self.wordIndexDic:
            return self.wordIndexDic[word]
        else:
            return self.wordIndexDic["UNK"]

vocabulary = Vocabulary()  # object to store vocabulary

train_data = []
with open("reuters/data/text.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        words = line.split() # get each word by splitting ' '

        # remove stop words and stem
        ps = PorterStemmer()
        words = [ps.stem(word) for word in words if not word in set(stopwords.words('english'))]
        train_data.append(words) # add to training data

        for word in words:  # add to vocabulary
            vocabulary.addToDicionary(word)

train_labels = []
with open("reuters/data/labels.txt", "r") as f:
    lines = f.readlines() # get line and remove trailing whitespaces
    for word in lines:
        word = word.rstrip()
        train_labels.append(labels[word])

test_data = []
with open("reuters/test/text.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        words = line.split() # get each word by splitting ' '

        # remove stop words and stem
        ps = PorterStemmer()
        words = [ps.stem(word) for word in words if not word in set(stopwords.words('english'))]
        test_data.append(words) # add to testing data

test_labels = []
with open("reuters/test/labels.txt", "r") as f:
    lines = f.readlines() # get line and remove trailing whitespaces
    for word in lines:
        word = word.rstrip()
        test_labels.append(labels[word])

# convert training and testing words into indexes
train_data[:] = [ [vocabulary.getIndex(w) for w in doc] for doc in train_data]
test_data[:] = [ [vocabulary.getIndex(w) for w in doc] for doc in test_data]

num_classes = 6
test_labels_old = test_labels  # for confusion matrix (added because of first variation)

# ------------ end of pre-processing


# ------------ variation number one - bag of words 
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
train_data = tokenizer.sequences_to_matrix(train_data, mode="binary")
test_data = tokenizer.sequences_to_matrix(test_data, mode="binary")

train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

model = models.Sequential()
model.add(layers.Dense(256, input_shape=(max_words,), activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(num_classes, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_data, train_labels, epochs=10, batch_size=32, verbose=1, validation_split=0.1, shuffle=True)
# ------------ end of variation one


# ------------ variation number two - using GRU
# train_data = np.asarray(sequence.pad_sequences(train_data, maxlen=100))
# test_data = np.asarray(sequence.pad_sequences(test_data, maxlen=100))

# model = models.Sequential()
# model.add(layers.Embedding(vocabulary.no_word, 128, input_length=100))
# model.add(layers.GRU(100, dropout=0.2))
# model.add(layers.Dense(num_classes, activation='softmax'))
# model.summary()

# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# history = model.fit(train_data, train_labels, epochs=25, batch_size=32, verbose=1, validation_split=0.1, shuffle=True)
# ------------ end of variation two


# ------------ variation number three - using LSM
# train_data = np.asarray(sequence.pad_sequences(train_data, maxlen=100))
# test_data = np.asarray(sequence.pad_sequences(test_data, maxlen=100))

# model = models.Sequential()
# model.add(layers.Embedding(vocabulary.no_word, 128, input_length=100))
# model.add(layers.LSTM(100, dropout=0.2))
# model.add(layers.Dense(num_classes, activation='softmax'))
# model.summary()

# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# history = model.fit(train_data, train_labels, epochs=25, batch_size=32, verbose=1, validation_split=0.1, shuffle=True)
# ------------ end of variation three


# ------------ evaluating final model and plotting results ---------------
def plotResults():
    # getting training time
    duration = round((timeit.default_timer() - start_time)/60, 1)

    # plotting error and accuracy
    # plot error
    # val_loss = validation, loss = training data
    plt.figure(num='Loss of Model')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model error')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper left')
    plt.show()

    # plot accuracy
    # val_loss = validation, loss = training data
    plt.figure(num='Accuracy of Model')
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper left')
    plt.show()

    # evaluating model on test set
    results = model.evaluate(test_data, test_labels, batch_size=32)
    print('test loss, test acc:', results)
    Y_pred = model.predict_classes(test_data)

    print('Confusion Matrix')
    print(confusion_matrix(test_labels_old, Y_pred))

    # get total number of parameters from the model
    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    total_params = trainable_count + non_trainable_count

    # plotting the confusion matrix
    (fig, ax) = plt.subplots(1, 1)
    ax.matshow(confusion_matrix(test_labels_old, Y_pred), cmap='bwr')
    ax.set_xlabel('output')
    ax.set_ylabel('target')
    ax.text(1.0, 0.5, 'Accuracy: {:.2%}\nDuration: {}min\nParams: {}'.format(results[1], duration, total_params),
            dict(fontsize=10, ha='left', va='center', transform=ax.transAxes))

    fig.canvas.set_window_title('Confusion matrix')
    fig.tight_layout()
    fig.show()

    input()

plotResults()