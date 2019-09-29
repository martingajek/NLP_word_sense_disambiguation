import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler
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
                torch.tensor(row['sent_indexes'][0], dtype=torch.int64), # Sentence encoding
                torch.tensor(row['target_token_idx'][0], dtype=torch.int64), # Target token indexes
                torch.tensor(row['is_proper_gloss'],dtype=torch.int64)) # Labels
    

class TrainValDataloader():
    """
    Class exposing train and validation dataloaders.
    each dataloader outputs 4 tensors:
     - the tensor of input token encodings
     - Sentence encoding tensor (1 and 0's)
     - Tensor of target token indexes
     - Binary label tensor

     the inputs are the fully sense-tagged tokenized and indexed corpus dataframe
     split in train data and test data. WHen  val_sample_dataloader is True
     classe exposes a subset validation set that is randomly sampled from the
     validation dataset. This serves to estimate validation error duing training
     

    other arguments to dataloaders:
    when pin_memory is True (it is used in conjunction with CUDA) to speed up memory transfer and
    increase GPU utilization
    num_workers same as pytorch dataloader class, num cpus

    the class exposes 2 dataloaders, namely the train_dataloader and val_dataloader    
    """
    def __init__(self, train_data, test_data, batch_size, val_sample_dataloader=False, 
                 val_sample_size=0.1,**kwargs):
        self.batch_size = batch_size
        
        self.train_df = train_data[['input_ids','sent_indexes',
                            'target_token_idx','is_proper_gloss']]
        self.val_df =  test_data[['input_ids','sent_indexes',
                            'target_token_idx','is_proper_gloss']]
        
        self.train_dataset = CorpusDataset(self.train_df)
        self.val_dataset = CorpusDataset(self.val_df)
        
        self.train_sampler = RandomSampler(self.train_dataset)
        self.val_sampler = SequentialSampler(self.val_dataset)
        
        self.train_dataloader = DataLoader(self.train_dataset, 
                                           sampler=self.train_sampler, 
                                           batch_size=self.batch_size,**kwargs)
    
        self.val_dataloader = DataLoader(self.val_dataset, 
                                         sampler=self.val_sampler, 
                                         batch_size=self.batch_size)

        if val_sample_dataloader:
            num_samples = int(len(self.val_dataset)*val_sample_size)
            assert num_samples > 0, "Number of samples for validation sample dataloader is 0, increase val_sample_size fraction" 
            self.subset_val_sampler = RandomSampler(self.val_dataset,num_samples=num_samples,replacement=True,)
            self.subset_val_dataloader = DataLoader(self.val_dataset, 
                                                 sampler=self.subset_val_sampler, 
                                                 batch_size=self.batch_size)



class TrainValSplitDataloader(TrainValDataloader):
    """
    Class exposing train and validation dataloaders.
    each dataloader outputs 4 tensors:
     - the tensor of input token encodings
     - Sentence encoding tensor (1 and 0's)
     - Tensor of target token indexes
     - Binary label tensor

     the inputs are the fully sense-tagged tokenized and indexed corpus dataframe
     as well as a batch size

    other arguments to dataloaders:
    when pin_memory is True (it is used in conjunction with CUDA) to speed up memory transfer and
    increase GPU utilization
    num_workers same as pytorch dataloader class, num cpus

    the class exposes 2 dataloaders, namely the train_dataloader and val_dataloader    
    """
    def __init__(self, data, batch_size, test_size=0.2, val_sample_dataloader=False, 
                 val_sample_size=0.1,**kwargs):
                
                data_subset = data[['input_ids','sent_indexes',
                            'target_token_idx','is_proper_gloss']]
                train_df, val_df =  train_test_split(data_subset, 
                                                       random_state=None, 
                                                       test_size=test_size)
                super().__init__(train_df, val_df, batch_size, val_sample_dataloader=val_sample_dataloader, 
                                val_sample_size=0.1,**kwargs)
        





