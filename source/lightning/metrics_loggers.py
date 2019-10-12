import numpy as np

class metrics_logger():
    def __init__(self,running_average=False):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.numsamples = 0
        self.running_average = running_average
        
        
    def update(self,_predicted,_labels):
        #ipdb.set_trace()
        _predicted= _predicted.to('cpu').numpy()
        _labels= _labels.to('cpu').numpy()
        pos_label_idx = _labels == 1
        if not self.running_average:
            self.reset()        
        if np.any(pos_label_idx):
            _positives = _predicted[pos_label_idx] == _labels[pos_label_idx]
            if self.running_average:
                self.TP += np.sum(_positives)
                self.FN += np.sum(~_positives)
            else:
                self.TP += np.sum(_positives)
                self.FN += np.sum(~_positives)
        neg_label_idx = _labels == 0
        if np.any(neg_label_idx):
            _negatives = _predicted[neg_label_idx] == _labels[neg_label_idx]
            self.TN += np.sum(_negatives)
            #ipdb.set_trace()
            self.FP += np.sum(~_negatives)       
        
    @property
    def accuracy(self):
        TP,TN = self.TP,self.TN
        FP,FN = self.FP,self.FN
        if TP+TN+FP+FN == 0:
            return np.NaN
        return (TP+TN)*1.0/(TP+TN+FP+FN)
    
    @property
    def precision(self):
        TP,TN = self.TP,self.TN
        FP,FN = self.FP,self.FN
        if TP == 0:
            return 0
        if TP+FP == 0:
            return np.NaN
        return TP*1.0/(TP+FP)

    @property
    def recall(self):
        TP,TN = self.TP,self.TN
        FP,FN = self.FP,self.FN
        if TP == 0:
            return 0
        if TP+FN == 0:
            return np.NaN
        return TP*1.0/(TP+FN)
    
    @property
    def f1(self):
        precision = self.precision
        recall = self.recall
        if precision+recall == 0:
            return np.NaN
        return 2*(precision*recall)/(precision+recall)
    
    def reset(self):
        self.TP,self.TN = 0,0
        self.FP,self.FN = 0,0