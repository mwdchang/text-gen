# Adapated from keras LSTM example
# https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

class CLSTM():
  def __init__(self, maxlen=40, step=3):
    self.maxlen = maxlen
    self.step = step 
    self.model = Sequential()
    self.epoch = 0


  ############################################################    
  ## Prepare input and model layers
  ############################################################    
  def setInput(self, fname):
    text = open(fname).read().lower()
    chars = sorted(list(set(text)))

    self.text = text
    self.chars = chars
    self.char_indices = dict((c, i) for i, c in enumerate(chars))
    self.indices_char = dict((i, c) for i, c in enumerate(chars))

    maxlen = self.maxlen
    sentences = []
    next_chars = []
    # cut the text in semi-redundant sequences of maxlen characters
    for i in range(0, len(text) - maxlen, self.step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])

    #print('Corpus length:', len(text))
    #print('Total chars:', len(chars))
    #print('nb sequences:', len(sentences))

    # Vectorize
    self.x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    self.y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            self.x[i, t, self.char_indices[char]] = 1
        self.y[i, self.char_indices[next_chars[i]]] = 1

    #print('x shape', np.shape(self.x))
    #print('y shape', np.shape(self.y))

    # Build model
    self.model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    self.model.add(Dense(len(chars)))
    self.model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)


  ############################################################    
  # Training
  ############################################################    
  def train(self):
    self.epoch = self.epoch + 1
    self.model.fit(self.x, self.y, batch_size=128, epochs=1)


  ############################################################    
  # Postproces single character sampling
  ############################################################    
  def sample(self, preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
     

  ############################################################    
  # Generate
  ############################################################    
  def generate_text(self, diversity):
    maxlen = self.maxlen
    chars = self.chars
    generated = ''
    start_index = random.randint(0, len(self.text) - self.maxlen - 1)
    seed = self.text[start_index: start_index + maxlen]
    generated += seed 

    for i in range(200):
      x_pred = np.zeros((1, maxlen, len(chars)))
      for t, char in enumerate(seed):
        x_pred[0, t, self.char_indices[char]] = 1.

      preds = self.model.predict(x_pred, verbose=0)[0]
      next_index = self.sample(preds, diversity)
      next_char = self.indices_char[next_index]

      generated += next_char
      seed = seed[1:] + next_char
    return generated
