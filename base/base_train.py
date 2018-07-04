# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:07:07 2018

@author: jwm
"""

class BaseTrain(object):
    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data
        

    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.max_nrof_epochs + 1, 1):
            loss = self.train_epoch(cur_epoch)
            self.sess.run(self.model.increment_cur_epoch_tensor)
            if cur_epoch > self.config.max_nrof_epochs or loss < 0.0001:
                break

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
