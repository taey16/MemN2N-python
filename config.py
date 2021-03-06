#!/usr/bin/python

import numpy as np

class BabiConfig(object):
  """
  Configuration for bAbI
  """
  def __init__(self, train_story, train_questions, dictionary):
    print('We will use BabiConfig')
    self.dictionary = dictionary
    self.batch_size = 32
    self.nhops = 3
    self.nepochs = 100
    self.lrate_decay_step = 25   # reduce learning rate by half every 25 epochs

    # Use 10% of training data for validation
    nb_questions = train_questions.shape[1]
    nb_train_questions = int(nb_questions * 0.9)

    self.train_range = np.array(range(nb_train_questions))
    self.val_range = np.array(range(nb_train_questions, nb_questions))
    self.enable_time = True   # add time embeddings
    self.use_bow = False  # use Bag-of-Words instead of Position-Encoding
    self.linear_start = True
    self.share_type = 1    # 1: adjacent, 2: layer-wise weight tying
    self.randomize_time = 0.1  # amount of noise injected into time index
    self.add_proj = False  # add linear layer between internal states
    self.add_nonlin = False  # add non-linearity to internal states

    self.display_inteval = 10

    if self.linear_start:
      self.ls_nepochs = 20
      self.ls_lrate_decay_step = 21
      self.ls_init_lrate = 0.01 / 2

    # Training configuration
    self.train_config = {
      "init_lrate" : 0.01,
      "max_grad_norm" : 40,
      "in_dim" : 20,
      "out_dim" : 20,
      "sz" : min(50, train_story.shape[1]),  # number of sentences
      "voc_sz" : len(self.dictionary),
      "bsz" : self.batch_size,
      "max_words" : len(train_story),
      "weight" : None
    }

    if self.linear_start:
      self.train_config["init_lrate"] = self.ls_init_lrate

    if self.enable_time:
      self.train_config.update({
        "voc_sz"   : self.train_config["voc_sz"] + self.train_config["sz"],
        "max_words": self.train_config["max_words"] + 1  # Add 1 for time words
       })


class BabiConfigJoint(object):
  """
  Joint configuration for bAbI
  """
  def __init__(self, train_story, train_questions, dictionary):
    print('We will use BabiConfigJoint')

    # TODO: Inherit from BabiConfig
    self.dictionary = dictionary
    self.batch_size = 32
    self.nhops = 4
    self.nepochs = 60 # 1K training samples
    #self.nepochs = 20 # 10K training samples

    self.lrate_decay_step = 15   # 1k training sampels reduce learning rate by half every 15 epochs 
    #self.lrate_decay_step = 5   # 10k training sampels reduce learning rate by half every 5 epochs

    # Use 10% of training data for validation
    nb_questions = train_questions.shape[1]
    nb_train_questions = int(nb_questions * 0.9)

    # Randomly split to training and validation sets
    rp = np.random.permutation(nb_questions)
    self.train_range = rp[:nb_train_questions]
    self.val_range = rp[nb_train_questions:]

    self.enable_time = True   # add time embeddings
    self.use_bow = False # use Bag-of-Words instead of Position-Encoding
    # we explored commencing training with the softmax in each memory layer removed, 
    # making the model entirely linear except for the final softmax for answer prediction. 
    # When the validation loss stopped decreasing, 
    # the softmax layers were re-inserted and training recommenced.
    self.linear_start = True
    self.share_type = 1    # 1: adjacent, 2: layer-wise weight tying
    self.randomize_time = 0.1  # amount of noise injected into time index (Random Noise RN)
    self.add_proj = False  # add linear layer between internal states
    self.add_nonlin = False  # add non-linearity to internal states

    self.display_inteval = 10

    if self.linear_start:
      self.ls_nepochs = 30
      self.ls_lrate_decay_step = 31
      self.ls_init_lrate = 0.01 / 2 # eta = 0.005

    # Training configuration
    self.train_config = {
      "init_lrate"   : 0.01, # NOTE: it is ignored instead ls_init_lrate when using LS
      "max_grad_norm": 40, # NOTE: clip gradient
      "in_dim"     : 50,  # NOTE: soft-attention memory slot(sentence) dim
      "out_dim"    : 50,  # NOTE: output memory slot(sentence) dim
      "sz" : min(50, train_story.shape[1]), # The capacity of memory is restricted to the most recent 50 sentences.
      "voc_sz"     : len(self.dictionary),
      "bsz"      : self.batch_size,
      "max_words"  : len(train_story),
      "weight"     : None
    }

    if self.linear_start:
      self.train_config["init_lrate"] = self.ls_init_lrate

    if self.enable_time:
      self.train_config.update({
        "voc_sz"   : self.train_config["voc_sz"] + self.train_config["sz"],
        "max_words": self.train_config["max_words"] + 1  # Add 1 for time words
       })


class Babi10kConfigJoint(object):
  """
  Joint configuration for bAbI-10k
  """
  def __init__(self, train_story, train_questions, dictionary):
    print('We will use Babi10kConfigJoint')

    self.dictionary = dictionary
    self.batch_size = 32
    self.nhops = 3
    self.nepochs = 100 # 1K training samples

    self.lrate_decay_step = 20  # 1k training sampels reduce learning rate by half every 15 epochs

    # Use 10% of training data for validation
    nb_questions = train_questions.shape[1]
    nb_train_questions = int(nb_questions * 0.9)

    # Randomly split to training and validation sets
    rp = np.random.permutation(nb_questions)
    self.train_range = rp[:nb_train_questions]
    self.val_range = rp[nb_train_questions:]

    self.enable_time = True   # add time embeddings
    self.use_bow = False  # use Bag-of-Words instead of Position-Encoding
    # we explored commencing training with the softmax in each memory layer removed, 
    # making the model entirely linear except for the final softmax for answer prediction. 
    # When the validation loss stopped decreasing, 
    # the softmax layers were re-inserted and training recommenced.
    self.linear_start = True
    self.share_type = 1    # 1: adjacent, 2: layer-wise weight tying (RNN-style)
    self.randomize_time = 0.1  # amount of noise injected into time index (Random Noise RN)
    self.add_proj = False  # add linear layer between internal states
    self.add_nonlin = False  # add non-linearity to internal states

    self.display_inteval = 10

    if self.linear_start:
      self.ls_nepochs = 5  # NOTE
      self.ls_lrate_decay_step = 6 # NOTE
      self.ls_init_lrate = 0.01 # NOTE: eta = 0.005

    # Training configuration
    self.train_config = {
      "init_lrate"   : 0.01, # NOTE: it is ignored instead ls_init_lrate when using LS
      "max_grad_norm": 40, # NOTE: clip gradient
      "in_dim"     : 50,  # NOTE: soft-attention memory slot(sentence) dim
      "out_dim"    : 50,  # NOTE: output memory slot(sentence) dim
      "sz" : min(50, train_story.shape[1]), # The capacity of memory is restricted to the most recent 50 sentences.
      "voc_sz"     : len(self.dictionary),
      "bsz"      : self.batch_size,
      "max_words"  : len(train_story),
      "weight"     : None
    }

    if self.linear_start:
      self.train_config["init_lrate"] = self.ls_init_lrate

    if self.enable_time:
      self.train_config.update({
        "voc_sz"   : self.train_config["voc_sz"] + self.train_config["sz"],
        "max_words": self.train_config["max_words"] + 1  # Add 1 for time words
      })
