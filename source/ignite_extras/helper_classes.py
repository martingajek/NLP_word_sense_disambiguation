#!/usr/bin/env python
# coding: utf-8
## Pytorch ignite helper classes

import torch
from torch.nn import functional as F

class Ignite_Engines():
    """
    wraps around and Instantiates several pytorch ignite functions,
    non_blocking argument is used to speed up memory transfer when using gpu 
    """
    def __init__(self,_model, _optimizer,_criterion,_device,non_blocking=False):
        self.model = _model
        self.optimizer = _optimizer
        self.criterion = _criterion
        self.device = _device
        self.non_blocking=non_blocking

    def get_process_function(self):
        def process_function(engine, batch):
            self.model.train()
            self.optimizer.zero_grad()
            batch = (tens.to(self.device,non_blocking=self.non_blocking) for tens in batch)
            b_tokens_tensor, b_sentence_tensor, b_target_token_tensor, y = batch
            y_pred = self.model(b_tokens_tensor, b_sentence_tensor, b_target_token_tensor)
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()
            return loss.item()
        return process_function

    def get_eval_function(self):
        def eval_function(engine, batch):
            self.model.eval()
            with torch.no_grad():
                batch = (tens.to(self.device,non_blocking=self.non_blocking) for tens in batch)
                b_tokens_tensor, b_sentence_tensor, b_target_token_tensor, y = batch
                logits = self.model(b_tokens_tensor, b_sentence_tensor, b_target_token_tensor)
                sm = F.softmax(logits,dim=1)
                y_pred = torch.argmax(sm,dim=1)
                return y_pred, y
        return eval_function
        
    def get_subset_eval_function(self):
        eval_function = self.get_eval_function()

        def subset_eval_function(engine, batch):
            """ Function ot be run on validation subset during the training process """
            y_pred, y = eval_function(engine, batch)
            with torch.no_grad():
                loss = self.criterion(y_pred, y)
                return loss.item()
        return subset_eval_function