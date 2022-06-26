'''
This object stores what comes out of each trial run. 
It stores the following:
- the model
- model trades
- trade returns
- trade results
'''

class ModelResult:

    def __init__(self, input_cols, model=None):
        self.input_cols = input_cols
        self.model = model
        self.kfolds = []

        self.train_losses, self.train_auc_rocs, self.train_best_thresholds = [], [], []
        self.valid_losses, self.valid_auc_rocs, self.valid_best_thresholds = [], [], []
        self.test_losses, self.test_auc_rocs, self.test_best_thresholds = [], [], []

        self.train_loss_avg, self.train_auc_roc_avg = -1, -1
        self.valid_loss_avg, self.valid_auc_roc_avg = -1, -1
        self.test_loss_avg, self.test_auc_roc_avg = -1, -1


    def describe(self):
        print(f'''
        -- results:
        -----
        --- train auc_roc per fold: {[round(x, 3) for x in self.train_auc_rocs]}
        --- valid auc_roc per fold: {[round(x, 3) for x in self.valid_auc_rocs]}
        --- test  auc_roc per fold: {[round(x, 3) for x in self.test_auc_rocs]}
        -----
        --- avg train auc_roc:      {round(self.train_auc_roc_avg, 3)}
        --- avg valid auc_roc:      {round(self.valid_auc_roc_avg, 3)}
        --- average test auc_roc:   {round(self.test_auc_roc_avg, 3)}
        -----
        --- avg train loss:         {round(self.train_loss_avg, 3)}
        --- avg valid loss:         {round(self.valid_loss_avg, 3 )}
        --- avg test loss:          {round(self.test_loss_avg, 3)}
        -----
        --- best test thresholds:   {[round(x, 3) for x in self.test_best_thresholds]}
        -----
        ''')


    def compute_averages(self):
        if not self.train_losses or not self.valid_losses or not self.test_losses:
            print("One of the loss lists is empty, exiting.")
            exit()
        elif not self.train_auc_rocs or not self.valid_auc_rocs or not self.test_auc_rocs:
            print("One of the auc_roc lists is empty, exiting.")
            exit()
        else:
            self.train_loss_avg = sum(self.train_losses) * 1.0 / float(len(self.train_losses))
            self.train_auc_roc_avg = sum(self.train_auc_rocs) / float(len(self.train_auc_rocs))

            self.valid_loss_avg = sum(self.valid_losses) * 1.0 / float(len(self.valid_losses))
            self.valid_auc_roc_avg = sum(self.valid_auc_rocs) / float(len(self.valid_auc_rocs))

            self.test_loss_avg = sum(self.test_losses) * 1.0 / float(len(self.test_losses))
            self.test_auc_roc_avg = sum(self.test_auc_rocs) / float(len(self.test_auc_rocs))


    def append_train(self, loss, auc_roc, threshold):
        self.train_losses.append(loss)
        self.train_auc_rocs.append(auc_roc)
        self.train_best_thresholds.append(threshold)

    
    def append_valid(self, loss, auc_roc, threshold):
        self.valid_losses.append(loss)
        self.valid_auc_rocs.append(auc_roc)
        self.valid_best_thresholds.append(threshold)


    def append_test(self, loss, auc_roc, threshold):
        self.test_losses.append(loss)
        self.test_auc_rocs.append(auc_roc)
        self.test_best_thresholds.append(threshold)


    def get_model(self):
        return self.model

    
    def get_input_cols(self):
        return self.input_cols


    def get_train_losses(self):
        return self.train_losses


    def get_train_auc_rocs(self):
        return self.train_auc_rocs


    def get_train_best_thresholds(self):
        return self.train_best_thresholds


    def get_valid_losses(self):
        return self.train_losses


    def get_valid_auc_rocs(self):
        return self.train_auc_rocs


    def get_valid_best_thresholds(self):
        return self.train_best_thresholds


    def get_test_losses(self):
        return self.train_losses


    def get_test_auc_rocs(self):
        return self.train_auc_rocs


    def get_test_best_thresholds(self):
        return self.train_best_thresholds

    
    def get_test_auc_roc_avg(self):
        return self.test_auc_roc_avg


    # only used to set a default value to be overriden (useful in for loop max / min)
    def set_test_auc_roc_avg(self, avg):
        self.test_auc_roc_avg = avg


    def set_model(self, model):
        self.model = model
