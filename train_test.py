from __future__ import division

import os
import sys
import math
import numpy as np

from memn2n.nn import Softmax
from util import Progress


def train(train_story, 
          train_questions, 
          train_qstory, 
          memory, 
          model, 
          loss, 
          general_config,
          train_logger,
          val_logger,
          global_batch_iter=0):

  train_config = general_config.train_config
  dictionary = general_config.dictionary
  nepochs = general_config.nepochs
  nhops = general_config.nhops
  batch_size = general_config.batch_size
  enable_time = general_config.enable_time
  randomize_time = general_config.randomize_time
  lrate_decay_step = general_config.lrate_decay_step

  train_range = general_config.train_range  # indices of training questions
  val_range = general_config.val_range  # indices of validation questions
  train_len = len(train_range)
  val_len = len(val_range)

  display_inteval = general_config.display_inteval

  params = {
    "lrate": train_config["init_lrate"],
    "max_grad_norm": train_config["max_grad_norm"]
  }

  if randomize_time > 0:
    print('We use Random Noise (RN) ratio of %.1f' % randomize_time)

  for ep in range(nepochs):
    # Decrease learning rate after every decay step
    if (ep + 1) % lrate_decay_step == 0:
      params["lrate"] *= 0.5

    total_err  = 0.
    total_cost = 0.
    total_num  = 0
    batch_iter = 0
    for _ in Progress(range(int(math.floor(train_len / batch_size)))):
      # Question batch
      batch = train_range[np.random.randint(train_len, size=batch_size)]

      input_data  = np.zeros((train_story.shape[0], batch_size), np.float32) # words of training questions
      target_data = train_questions[2, batch]                # indices of training answers

      memory[0].data[:] = dictionary["nil"]

      # Compose batch of training data
      for b in range(batch_size):
        # NOTE: +1 since train_questions[1, :] is the index of the sentence right before the training question.
        # d is a batch of [word indices in sentence, sentence indices from batch] for this story
        d = train_story[:, :(1 + train_questions[1, batch[b]]), train_questions[0, batch[b]]]

        # Pick a fixed number of latest sentences (before the question) from the story
        offset = max(0, d.shape[1] - train_config["sz"])
        d = d[:, offset:]

        # Training data for the 1st memory cell
        memory[0].data[:d.shape[0], :d.shape[1], b] = d

        if enable_time:
          # Inject noise into time index (i.e. word index)
          if randomize_time > 0:
            # Random number of blank (must be < total sentences until the training question?)
            nblank = np.random.randint(int(math.ceil(d.shape[1] * randomize_time)))
            rt = np.random.permutation(d.shape[1] + nblank)

            rt[rt >= train_config["sz"]] = train_config["sz"] - 1 # put the cap

            # Add random time (must be > dictionary's length) into the time word (decreasing order)
            memory[0].data[-1, :d.shape[1], b] = np.sort(rt[:d.shape[1]])[::-1] + len(dictionary)

          else:
            memory[0].data[-1, :d.shape[1], b] = \
              np.arange(d.shape[1])[::-1] + len(dictionary)

        input_data[:, b] = train_qstory[:, batch[b]]

      for i in range(1, nhops):
        memory[i].data = memory[0].data

      out = model.fprop(input_data)
      cost = loss.fprop(out, target_data)
      err  = loss.get_error(out, target_data)
      total_cost += cost
      total_err  += err
      total_num  += batch_size

      grad = loss.bprop(out, target_data)
      model.bprop(input_data, grad)
      model.update(params)
      batch_iter += 1

      global_batch_iter += 1

      if batch_iter % display_inteval == 0:
        print("%d | %d | %g | loss: %g | err: %g" % \
               (ep, global_batch_iter, params['lrate'], cost / batch_size, err / batch_size))
        sys.stdout.flush()
        train_logger.write('%d %d %f %f %f\n' %(ep, global_batch_iter, params['lrate'], cost/batch_size, err/batch_size))
        train_logger.flush()

      for i in range(nhops):
        memory[i].emb_query.weight.D[:, 0] = 0

    # Validation
    total_val_err  = 0.
    total_val_cost = 0.
    total_val_num  = 0
    best_val_cost = 1000000.

    for k in range(int(math.floor(val_len / batch_size))):
      batch     = val_range[np.arange(k * batch_size, (k + 1) * batch_size)]
      input_data  = np.zeros((train_story.shape[0], batch_size), np.float32)
      target_data = train_questions[2, batch]

      memory[0].data[:] = dictionary["nil"]

      for b in range(batch_size):
        d = train_story[:, :(1 + train_questions[1, batch[b]]), train_questions[0, batch[b]]]

        offset = max(0, d.shape[1] - train_config["sz"])
        d = d[:, offset:]

        # Data for the 1st memory cell
        memory[0].data[:d.shape[0], :d.shape[1], b] = d

        if enable_time:
          memory[0].data[-1, :d.shape[1], b] = np.arange(d.shape[1])[::-1] + len(dictionary)

        input_data[:, b] = train_qstory[:, batch[b]]

      for i in range(1, nhops):
        memory[i].data = memory[0].data

      out = model.fprop(input_data)
      val_cost = loss.fprop(out, target_data)
      val_err  = loss.get_error(out, target_data)
      total_val_cost += val_cost
      total_val_err += val_err
      total_val_num += batch_size

    if best_val_cost > total_val_cost:
      best_model = model
      best_memory = memory
      best_loss = val_cost / total_val_num
      best_err = total_val_err / total_val_num
      print('Best loss: %f Best err: %f' % (best_loss, best_err))
      sys.stdout.flush()

    train_error = total_err / total_num
    val_error   = total_val_err / total_val_num

    print("%d | %d | loss: %g | err: %g" % (ep, global_batch_iter, total_val_cost / total_val_num, total_val_err / total_val_num))
    sys.stdout.flush()
    val_logger.write('%d %d %f %f %f\n' %(ep, global_batch_iter, params['lrate'], total_val_cost / total_val_num, total_val_err/total_val_num))
    val_logger.flush()

  return train_logger, val_logger, best_model, best_memory, global_batch_iter


def train_linear_start(train_story, 
                       train_questions, 
                       train_qstory, 
                       memory, 
                       model, 
                       loss, 
                       general_config, 
                       log_path='./'):

  train_config = general_config.train_config

  # Remove softmax from memory
  for i in range(general_config.nhops):
    memory[i].mod_query.modules.pop()

  # Save settings
  nepochs2 = general_config.nepochs
  lrate_decay_step2 = general_config.lrate_decay_step
  init_lrate2 = train_config["init_lrate"]

  # Add new settings
  general_config.nepochs = general_config.ls_nepochs
  general_config.lrate_decay_step = general_config.ls_lrate_decay_step
  train_config["init_lrate"] = general_config.ls_init_lrate

  train_logger = open(os.path.join(log_path, 'train.log'), 'w')
  train_logger.write('epoch batch_iter lr loss err\n')
  train_logger.flush()
  val_logger = open(os.path.join(log_path, 'val.log'), 'w')
  val_logger.write('epoch batch_iter lr loss err\n')
  val_logger.flush()

  global_batch_iter = 0
  # Train with new settings
  train_logger, val_logger, best_model, best_memory, global_batch_iter = \
    train(train_story, 
          train_questions, 
          train_qstory, 
          memory, 
          model, 
          loss, 
          general_config,
          train_logger, 
          val_logger,
          global_batch_iter)

  # When the validation loss stopped decreasing, 
  # the softmax layers were re-inserted and training recommenced.
  # Add softmax back
  for i in range(general_config.nhops):
    memory[i].mod_query.add(Softmax())

  # Restore old settings
  general_config.nepochs = nepochs2
  general_config.lrate_decay_step = lrate_decay_step2
  train_config["init_lrate"] = init_lrate2

  # Train with old settings
  train_logger, val_logger, best_model, best_memory, _ = \
    train(train_story, 
          train_questions, 
          train_qstory, 
          memory, 
          model, 
          loss, 
          general_config, 
          train_logger, 
          val_logger,
          global_batch_iter)

  train_logger.close()
  val_logger.close()

  return best_model, best_memory


def test(test_story, 
         test_questions, 
         test_qstory, 
         memory, 
         model, 
         loss, 
         general_config):
  total_test_err = 0.
  total_test_num = 0

  nhops = general_config.nhops
  train_config = general_config.train_config
  batch_size = general_config.batch_size
  dictionary = general_config.dictionary
  enable_time = general_config.enable_time

  max_words = train_config["max_words"] \
    if not enable_time else train_config["max_words"] - 1

  for k in range(int(math.floor(test_questions.shape[1] / batch_size))):
    batch = np.arange(k * batch_size, (k + 1) * batch_size)

    input_data = np.zeros((max_words, batch_size), np.float32)
    target_data = test_questions[2, batch]

    input_data[:]   = dictionary["nil"]
    memory[0].data[:] = dictionary["nil"]

    for b in range(batch_size):
      d = test_story[:,:(1 + test_questions[1, batch[b]]), test_questions[0, batch[b]]]

      offset = max(0, d.shape[1] - train_config["sz"])
      d = d[:, offset:]

      memory[0].data[:d.shape[0], :d.shape[1], b] = d

      if enable_time:
        memory[0].data[-1, :d.shape[1], b] = np.arange(d.shape[1])[::-1] + len(dictionary) # time words

      input_data[:test_qstory.shape[0], b] = test_qstory[:, batch[b]]

    for i in range(1, nhops):
      memory[i].data = memory[0].data

    out = model.fprop(input_data)
    # cost = loss.fprop(out, target_data)
    total_test_err += loss.get_error(out, target_data)
    total_test_num += batch_size

  test_error = total_test_err / total_test_num
  print("Test error: %f" % test_error)
