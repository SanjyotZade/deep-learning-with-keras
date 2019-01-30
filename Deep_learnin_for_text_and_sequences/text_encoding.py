import numpy as np

samples  = ["The cat sat on the mat"," The dog ate my homework"]

token_index = {}


# Encoding words in the corpus
for sample in samples:
    for word in sample.split():
        if word not in token_index.keys():
            token_index[word] = len(token_index) + 1


# Word level one hot encoding
max_length = 10
results = np.zeros(shape=(len(samples),max_length,
                   len(token_index)+1))

for i , sample in enumerate(samples):
    for j , word in enumerate(sample.split()):
        index_ = token_index[word]
        results[i,j,index_] = 1

print results


# Letter level one hot encoding
import string
characters = string.printable

token_index =  dict(zip((range(1,len(characters)+1)),characters))
max_length  = 50

results = np.zeros(shape=(len(samples),max_length,len(characters)+1))


for i , sample in enumerate(samples):
    for j , character in enumerate(samples):
        results[i,j,token_index.get(character)] = 1

# One-hot encoding using keras
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=10)
tokenizer.fit_on_texts(samples)

sequences = tokenizer.texts_to_sequences(samples)
one_hot_results = tokenizer.texts_to_matrix(samples,mode='binary')


word_index = tokenizer.word_index


print one_hot_results