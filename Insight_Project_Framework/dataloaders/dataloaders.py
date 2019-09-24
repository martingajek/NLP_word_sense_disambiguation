import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch


class CorpusDataset(Dataset):
    """
    pytorch dataset handling class    
    """
    def __init__(self, data):
        self.corpus_dataframe = data

    def __len__(self):
        return self.corpus_dataframe.shape[0]

    def __getitem__(self, idx):
        row = self.corpus_dataframe.iloc[idx]
        return (torch.tensor(row['input_ids'][0]),  # Input token encodings
                torch.tensor(row['sent_indexes'][0]), # Sentence encoding
                torch.tensor(row['target_token_idx'][0]), # Target token indexes
                torch.tensor(row['is_proper_context'],dtype=torch.float)) # Labels
    
class TrainValDataloader():
    """
    Class exposing train and validation dataloaders.
    each dataloader outputs 4 tensors:
     - the tensor of input token encodings
     - Sentence encoding tensor (1 and 0's)
     - Tensor of target token indexes
     - Binary label tensor

     the inputs are the fully sense-tagged tokenized and indexed corpus dataframe
     as well as a batch size

    the class exposes 2 dataloaders, namely the train_dataloader and val_dataloader    
    """
    def __init__(self, data, batch_size, test_size=0.2):
        self.batch_size = batch_size
        data_subset = data[['input_ids','sent_indexes',
                            'target_token_idx','is_proper_context']]
        self.train_df, self.val_df =  train_test_split(data_subset, 
                                                       random_state=None, 
                                                       test_size=test_size)
        
        self.train_dataset = CorpusDataset(self.train_df)
        self.val_dataset = CorpusDataset(self.val_df)
        
        self.train_sampler = RandomSampler(self.train_dataset)
        self.val_sampler = SequentialSampler(self.val_dataset)
        
        self.train_dataloader = DataLoader(self.train_dataset, 
                                           sampler=self.train_sampler, 
                                           batch_size=self.batch_size)
    
        self.val_dataloader = DataLoader(self.val_dataset, 
                                         sampler=self.val_sampler, 
                                         batch_size=self.batch_size)
    