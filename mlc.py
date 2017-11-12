# Adapted from
# http://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139

#
# Maximum likelyhood character level language modelling


from collections import *
from random import random
import sys


def normalize(counter):
  s = float(sum(counter.values()))
  return [(c,cnt/s) for c,cnt in counter.iteritems()]


class MLC():
  def __init__(self, order=4):
    self.order = order

  def train_fulltext(self, fname):
    data = file(fname).read()
    lm = defaultdict(Counter)
    pad = "~" * self.order
    data = pad + data 
    for i in xrange(len(data) - self.order):
      history, char = data[i: i + self.order], data[i + self.order]
      lm[history][char]+=1
    outlm = {hist:normalize(chars) for hist, chars in lm.iteritems()}
    # print("Model size > " + str(len(outlm)))
    self.lm = outlm


  def train_lines(self, fname):
    data = file(fname).read()
    lm = defaultdict(Counter)
    pad = "~" * self.order

    sentences = data.split('\n')
    for sent in sentences:
      sent = pad + sent
      for i in xrange(len(sent) - self.order):
        history, char = sent[i: i + self.order], sent[i + self.order]
        lm[history][char]+=1

    outlm = {hist:normalize(chars) for hist, chars in lm.iteritems()}
    # print("Model size > " + str(len(outlm)))
    self.lm = outlm

  def generate_letter(self, history):
    history = history[-self.order:]

    # Signals end
    if self.lm.get(history) is None:
      return None

    dist = self.lm[history]
    x = random()
    for c,v in dist:
      x = x - v
      if x <= 0: return c

  def generate_text(self, nletters=200, history = None):
    if history is None:
      history = "~" * self.order
    else:
      history = "~" * (self.order - len(history)) + history
      history = history[-self.order:]
    start = history

    out = []
    for i in xrange(nletters):
      c = self.generate_letter(history) 
      if c is None:
        break
      history = history[-self.order:] + c
      out.append(c)
    return start + "".join(out)
