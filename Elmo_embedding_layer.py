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

class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

def build(self, input_shape):
    self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable, name="{}_module".format(self.name))
    self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
    super(ElmoEmbeddingLayer, self).build(input_shape)

def call(self, x, mask=None):
    result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                       as_dict=True,
                       signature='default',
                       )['default']
    return result
def compute_mask(self, inputs, mask=None):
    return K.not_equal(inputs, '--PAD--')
def compute_output_shape(self, input_shape):
    return (input_shape[0], self.dimensions)
