
#Model Imports
from keras.datasets import reuters
from keras import layers
from keras import models
import numpy as np
import matplotlib.pyplot as plt


def decode_train_sample(sample):
    word_index = reuters.get_word_index()
    reverse_word_index = dict([(index, word) for word, index in word_index.items()])
    return " ".join([ reverse_word_index.get(i-3,"?") for i in sample])

def vectorize_sequences(sequence,dimension = 10000):
    vector = np.zeros((len(sequence),dimension))
    for index,word_num in enumerate(sequence):
        vector[index,word_num] = 1.
    return vector

def one_hot_encoding(labels,dimension=46):
    one_hot_vector = np.zeros((len(labels),dimension))
    for index,label_num in enumerate(labels):
        one_hot_vector[index,label_num] = 1.
    return one_hot_vector

def plot_graphs(parameter1,parameter2):

    loss = model_history.history[parameter1]
    val_loss = model_history.history[parameter2]

    epocs = range(1, len(loss) + 1)

    plt.plot(epocs, loss, 'bo', label=parameter1)
    plt.plot(epocs, val_loss, 'b', label=parameter2)
    plt.title('{0} and {1} loss'.format(parameter1,parameter2))
    plt.xlabel("Epocs")
    plt.ylabel(parameter1)
    plt.legend()
    plt.show()

# load reuter dataset
(train_data , train_labels) , (test_data , test_labels) = reuters.load_data(num_words=10000)

# print train_data.shape
# print test_data.shape
# print decode_train_sample(sample=train_data[0])


#Encode the data
x_train = vectorize_sequences(sequence=train_data)
x_test = vectorize_sequences(sequence=test_data)

one_hot_train_labels = one_hot_encoding(train_labels)
one_hot_test_labels = one_hot_encoding(test_labels)


#Starting to built the model
model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#Separating data into test and train
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

#Starting to train the model

model_history = model.fit(partial_x_train,
                          partial_y_train,
                          epochs=20,
                          batch_size=512,
                          validation_data=(x_val,y_val))


# plot_graphs(parameter1='loss',parameter2='val_loss')
# plot_graphs(parameter1='acc',parameter2='val_acc')

#Doing predictions

results = model.evaluate(x_test,one_hot_test_labels)
print results