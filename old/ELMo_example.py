from Elmo_embedding_layer import ElmoEmbeddingLayer

import pandas as pd
import json
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import keras.layers as layers
from keras.models import Model, load_model
from keras.engine import Layer
import tensorflow as tf

from sacremoses import MosesTokenizer, MosesDetokenizer
from allennlp.modules.elmo import Elmo, batch_to_ids


with open('yelp_academic_dataset_review.json') as json_file:
    data = json_file.readlines()
    data = list(map(json.loads, data))
    data = data[0:100000]

data_frame = pd.DataFrame(data)
nltk.download('wordnet')
print("loaded")

mt = MosesTokenizer()

def clean_review(text):
    # Strip HTML tags
    text = re.sub('<[^<]+?>', ' ', text)
    # Strip escaped quotes
    text = text.replace('\\"', '')
    # Strip quotes
    text = text.replace('"', '')
    return text

data_frame['cleaned_review'] = data_frame['text'].apply(clean_review)
# tokenizes each review
data_frame['tokenized_review'] = data_frame['cleaned_review'].apply(mt.tokenize)

X_train, X_test, y_train, y_test = train_test_split(data_frame['tokenized_review'], data_frame['stars'], test_size=0.2)

X_train = np.array(X_train, dtype=object)[:, np.newaxis]
X_test = np.array(X_test, dtype=object)[:, np.newaxis]

y_train = np.array(y_train)
y_test = np.array(y_test)

y_train = y_train - 1
y_test = y_test - 1

input_text = layers.Input(shape=(1,), dtype=tf.string)
embedding = ElmoEmbeddingLayer()(input_text)
dense = layers.Dense(256, activation='relu')(embedding)
dense2 = layers.Dense(256, activation='relu')(dense)
pred = layers.Dense(5, activation='sigmoid')(dense2)
model = Model(inputs=[input_text], outputs=pred)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X_train,
          y_train,
          validation_data=(X_train, y_train),
          epochs=5,
          batch_size=32)

scores = model.evaluate(X_test, y_test, verbose=1)
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
