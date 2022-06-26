'''
This class is used to train / test / validate a model on a data loader object
'''
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn, torch.cuda, torch.optim
import torch
import time
from constants.constants import SAVED_MODELS_PATH

from stats.stat_util import get_roc_data

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

class BaseModelTrain():
    
    def __init__(self, network, nn_config, input_cols, train_loader, valid_loader, test_loader, logs_enabled=False, loss_ratio=1):
        self.network = network
        self.nn_config = nn_config
        self.input_cols = input_cols

        self.best_model = copy.deepcopy(network)
        self.loss_ratio = loss_ratio

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=nn_config['learning_rate'])
   
        self.best_epoch = -1 # TODO make this useful (reload best model, etc)
        self.logs_enabled = logs_enabled
        self.filename = 'nn_1.0' + '_'.join(input_cols)

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader



    def main(self):
        print("--- starting training...")

        
        # setup vars for storing what are essentially training debug logs for later (if needed)
        self.min_train_loss, self.min_valid_loss, self.min_test_loss = 9999, 9999, 9999
        self.train_losses, self.valid_losses, self.test_losses = [], [], []

        self.train_acc, self.train_acc_pos, self.train_acc_neg = [], [], []
        self.valid_acc, self.valid_acc_pos, self.valid_acc_neg = [], [], []
        self.test_acc, self.test_acc_pos, self.test_acc_neg = [], [], []

        self.train_true_pos, self.train_false_pos, self.train_true_neg, self.train_false_neg = [], [], [], []
        self.valid_true_pos, self.valid_false_pos, self.valid_true_neg, self.valid_false_neg = [], [], [], []
        self.test_true_pos, self.test_false_pos, self.test_true_neg, self.test_false_neg = [], [], [], []

        
        train_time_start = time.time()
        patience = 0

        for epoch in range(1, self.nn_config['epochs'] + 1):
            start_time = time.time()

            self.train()
            self.validate()
            
            # self.network.load_state_dict(torch.load(best_model_file.name))
            
            elapsed_time = time.time() - start_time

            if self.logs_enabled:
                print(f"----------Epoch {epoch} + (time={round(elapsed_time, 2)}s aka {round(elapsed_time / 60, 2)} mins) ----------")
                print(f"TRAIN set: net loss = {self.train_losses[-1]}")
                print(f"TEST set:  net loss = {self.train_losses[-1]}\n")

            if self.valid_losses[-1] < self.min_valid_loss:
                self.min_loss_valid = self.valid_losses[-1]
                patience = 0
                self.best_epoch = epoch
                # self.__save_weights(filename)
                # print(f'Epoch {epoch} --- new best model: valid loss = {self.valid_losses[-1]} \n')
            else:
                patience += 1
                
            if patience == self.nn_config['patience_limit']:
                # print(f'Breaking on patience = {patience} ')
                # should prob reload best model here
                break


        if self.logs_enabled:
            self.__result_summary()

        train_time_elapsed = round(time.time() - float(train_time_start), 2)
        print(f'--- training complete ({train_time_elapsed} s)')


    def train(self, eval=False):
        self.network.train()
        total_loss = 0
        total_true_pos, total_false_pos, total_true_neg, total_false_neg = 0, 0, 0, 0
        
        for batch_idx, (inputs, label) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            inputs, label = inputs.float(), label.unsqueeze(1).float()
            output = self.network(inputs)

            loss = self.__weighted_bce_loss(output, label)
            loss.backward()
            self.optimizer.step()

            pred = output.clone().detach().clone().cpu()
            label = label.cpu()
            
            total_loss += loss.clone().detach().cpu().item()
            
            # TODO: next line should break? maybe? maybe not?
            true_pos, false_pos, true_neg, false_neg = self.__hard_binary_accuracy(pred, label)
            total_true_pos += true_pos
            total_false_pos += false_pos
            total_true_neg += true_neg
            total_false_neg += false_neg       
        
        train_acc = (total_true_pos + total_true_neg) / (total_true_pos + total_true_neg + total_false_pos + total_false_neg)
        train_acc_pos = total_true_pos / (total_true_pos + total_false_neg)
        train_acc_neg = total_true_neg / (total_true_neg + total_false_pos)

        self.train_acc.append(train_acc)
        self.train_acc_pos.append(train_acc_pos)
        self.train_acc_neg.append(train_acc_neg)
        
        net_loss = total_loss / (self.nn_config['batch_size'] * len(self.train_loader))
        self.train_losses.append(net_loss)

        self.train_true_pos.append(total_true_pos)
        self.train_false_pos.append(total_false_pos)
        self.train_true_neg.append(total_true_neg)
        self.train_false_neg.append(total_false_neg)

        if eval:
            tpr, fpr, roc_auc, best_threshold = get_roc_data(self.train_loader, self.network)
            return net_loss, roc_auc, best_threshold
    

    def validate(self, eval=False):
        self.network.eval()
        total_loss = 0    

        total_true_pos, total_false_pos, total_true_neg, total_false_neg = 0, 0, 0, 0
        
        with torch.no_grad():
            for batch_idx, (inputs, label) in enumerate(self.valid_loader):
                
                # TODO: same here
                # if torch.cuda.is_available:
                #     data, label = data.cuda(), label.cuda()
                
                inputs, label = inputs.float(), label.unsqueeze(1).float()
                output = self.network(inputs)
                
                criterion = torch.nn.BCELoss()
                total_loss += criterion(output, label).item()

                pred = output.detach().clone().cpu()
                label = label.cpu()
                
                true_pos, false_pos, true_neg, false_neg = self.__hard_binary_accuracy(pred, label)
                total_true_pos += true_pos
                total_false_pos += false_pos
                total_true_neg += true_neg
                total_false_neg += false_neg
        
        valid_acc = (total_true_pos + total_true_neg) / (total_true_pos + total_true_neg + total_false_pos + total_false_neg)
        valid_acc_pos = total_true_pos / (total_true_pos + total_false_neg)
        valid_acc_neg = total_true_neg / (total_true_neg + total_false_pos)

        self.valid_acc.append(valid_acc)
        self.valid_acc_pos.append(valid_acc_pos)
        self.valid_acc_neg.append(valid_acc_neg)

        net_loss = total_loss / (self.nn_config['batch_size'] * len(self.valid_loader))
        self.valid_losses.append(net_loss)
        
        self.valid_true_pos.append(total_true_pos)
        self.valid_false_pos.append(total_false_pos)
        self.valid_true_neg.append(total_true_neg)
        self.valid_false_neg.append(total_false_neg)
        
        if eval:
            tpr, fpr, roc_auc, best_threshold = get_roc_data(self.valid_loader, self.network)
            return net_loss, roc_auc, best_threshold


    def test(self):
        self.network.eval()
        total_loss = 0    

        total_true_pos, total_false_pos, total_true_neg, total_false_neg = 0, 0, 0, 0
        
        with torch.no_grad():
            for batch_idx, (inputs, label) in enumerate(self.test_loader):
                inputs, label = inputs.float(), label.unsqueeze(1).float()
                output = self.network(inputs)
                
                criterion = torch.nn.BCELoss()
                total_loss += criterion(output, label).item()

                pred = output.detach().clone().cpu()
                label = label.cpu()
                
                true_pos, false_pos, true_neg, false_neg = self.__hard_binary_accuracy(pred, label)
                total_true_pos += true_pos
                total_false_pos += false_pos
                total_true_neg += true_neg
                total_false_neg += false_neg
        
        test_acc = (total_true_pos + total_true_neg) / (total_true_pos + total_true_neg + total_false_pos + total_false_neg)
        test_acc_pos = total_true_pos / (total_true_pos + total_false_neg)
        test_acc_neg = total_true_neg / (total_true_neg + total_false_pos)

        self.test_acc.append(test_acc)
        self.test_acc_pos.append(test_acc_pos)
        self.test_acc_neg.append(test_acc_neg)

        net_loss = total_loss / (self.nn_config['batch_size'] * len(self.test_loader)) # I suppose we care about this
        self.test_losses.append(net_loss)
        
        self.test_true_pos.append(total_true_pos)
        self.test_false_pos.append(total_false_pos)
        self.test_true_neg.append(total_true_neg)
        self.test_false_neg.append(total_false_neg)

        tpr, fpr, roc_auc, best_threshold = get_roc_data(self.test_loader, self.network)
        return net_loss, roc_auc, best_threshold


    def __hard_binary_accuracy(self, batch, labels):
        batch = torch.round(batch)
        confusion_matrix = batch / labels
        """ Returns the confusion matrix for the values in the `prediction` and `truth`
        tensors, i.e. the amount of positions where the values of `prediction`
        and `truth` are
        - 1 and 1 (True Positive)
        - 1 and 0 (False Positive)
        - 0 and 0 (True Negative)
        - 0 and 1 (False Negative)
        """
        true_positives = torch.sum(confusion_matrix == 1).item()
        false_positives = torch.sum(confusion_matrix == float('inf')).item()
        true_negatives = torch.sum(torch.isnan(confusion_matrix)).item()
        false_negatives = torch.sum(confusion_matrix == 0).item()

        return true_positives, false_positives, true_negatives, false_negatives
    
    
    def __result_summary(self):
        # should be best not last, fix in future
        print(f'Final train loss: {self.train_losses[-1]}')
        print(f'Final valid loss:  {self.valid_losses[-1]}')
        print('---')
        print(f'Train set accuracy: {self.train_acc[-1]}')
        print(f'Pct good trades placed: {self.train_acc_pos[-1]}')
        print(f'Pct bad trades skipped: {self.train_acc_neg[-1]}')
        print('---')
        print(f'Valid set accuracy: {self.valid_acc[-1]}')
        print(f'Pct good trades placed: {self.valid_acc_pos[-1]}')
        print(f'Pct bad trades skipped: {self.valid_acc_neg[-1]}')
    

    def result_summary_test(self):
        print('---')
        print(f'Test set accuracy: {self.test_acc[-1]}')
        print(f'Pct good trades placed: {self.test_acc_pos[-1]}')
        print(f'Pct bad trades skipped: {self.test_acc_neg[-1]}')


    def __weighted_bce_loss(self, output, label): # punish positive case failures 4x for now
        criterion = torch.nn.BCELoss(reduction='none')
        loss = criterion(output, label)

        # for i in range(len(loss)):
        #     if label[i] == 1:
        #         loss[i] *= self.loss_ratio

        return torch.mean(loss)


    def __load_weights(self, filename):
        path = os.path.join(SAVED_MODELS_PATH, filename)
        try:
            self.network.load_state_dict(torch.load(path, verbose=False))
            print('successfully loaded')
        except:
            print('issue loading model weights')

    
    def __save_weights(self, filename):
        path = os.path.join(SAVED_MODELS_PATH, filename)
        torch.save(self.network.state_dict(), path)


    def plot_loss(self):
        plt.plot(self.train_losses, color='r')
        plt.plot(self.valid_losses, color='b')
        plt.show()