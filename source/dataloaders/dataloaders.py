import pandas as pd
import torch
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler
from dataloaders.transforms import transform
from dataloaders import data_format_utils as dfu


MAX_LEN = 128

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
    

class CorpusTransformDataset(CorpusDataset):
    """
    pytorch dataset handling class that handles the raw dataset transformation
    into bert embeddings
    """
    def __init__(self, data,pad_len=MAX_LEN,weak_supervision=False,tokenizer=dfu.DEF_TOKENIZER):
        super().__init__(data)
        self.pad_len = pad_len
        self.weak_super = weak_supervision
        self.tokenizer = tokenizer
    
    def __getitem__(self, idx):
        raw_row = self.corpus_dataframe.iloc[idx]
        labels = torch.tensor(raw_row['is_proper_gloss'],dtype=torch.int64)
        batch = transform(raw_row,self.pad_len,weak_supervision=self.weak_super,
                          tokenizer=self.tokenizer)
        # if the index is out of bounds default to 0 index
        # chich is the cls token in BERT
        if torch.gt(batch[2],self.pad_len).item():
            token_idx,sent_embed,target_idx = batch
            target_idx = torch.tensor(0,dtype=torch.int64)
            batch = tuple([token_idx,sent_embed,target_idx])
       
        return (*batch, # Target token indexes
                labels) # Labels

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
    def __init__(self, train_data, test_data, batch_size, val_dataloader=True, 
                 val_sample_size=0.1,pad_len=MAX_LEN,weak_supervision=False,tokenizer=dfu.DEF_TOKENIZER,
                 train_samples=None,
                 **kwargs):
        self.batch_size = batch_size
        
        self.train_df = train_data
        self.test_df = test_data

        
        self.train_dataset = CorpusTransformDataset(self.train_df,pad_len=pad_len,
                                                    weak_supervision=weak_supervision,
                                                    tokenizer=tokenizer)
        self.test_dataset = CorpusTransformDataset(self.test_df,pad_len=pad_len,
                                                 weak_supervision=weak_supervision,
                                                 tokenizer=tokenizer)
        
        if train_samples:
            self.train_sampler = RandomSampler(self.train_dataset,num_samples=train_samples,replacement=True)
        else:
            self.train_sampler = RandomSampler(self.train_dataset)
        self.test_sampler = SequentialSampler(self.test_dataset)
        
        self.train_dataloader = DataLoader(self.train_dataset, 
                                           sampler=self.train_sampler, 
                                           batch_size=self.batch_size,**kwargs)
    
        self.test_dataloader = DataLoader(self.test_dataset, 
                                         sampler=self.test_sampler, 
                                         batch_size=self.batch_size)

        if val_dataloader:
            num_samples = int(len(self.test_dataset)*val_sample_size)
            assert num_samples > 0, "Number of samples for validation sample dataloader is 0, increase val_sample_size fraction" 
            self.val_sampler = RandomSampler(self.test_dataset,
                                                    num_samples=num_samples,
                                                    replacement=True,)
            self.val_dataloader = DataLoader(self.test_dataset, 
                                                 sampler=self.val_sampler, 
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
    def __init__(self, data, batch_size, test_size=0.2, val_dataloader=False, 
                 val_sample_size=0.1,**kwargs):
                
                data_subset = data
                train_df, test_df =  train_test_split(data_subset, 
                                                       random_state=None, 
                                                       test_size=test_size)
                super().__init__(train_df, test_df, batch_size, val_dataloader=val_dataloader, 
                                val_sample_size=0.1,**kwargs)
        





