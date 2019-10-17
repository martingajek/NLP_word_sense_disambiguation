import pandas as pd
import os
from pytorch_transformers import BertTokenizer, XLNetTokenizer
from dataloaders.data_format_utils import preprocess_model_inputs
from dataloaders.dataloaders import  TrainValDataloader, TrainValSplitDataloader

#########################################################################################
#                                                                                       #
# Helper functions that takes in data path and spits out appropriate dataloader class   #
#                                                                                       #
#########################################################################################

os.path

def read_data_to_dataframe(_path,**kwargs):
    """
    Helper function that reads csv and feather formats and outputs
    pandas dataframe  
    """
    if _path.lower().endswith('.csv'):
        print('csv!')
        return pd.read_csv(_path,**kwargs)
    elif _path.lower().endswith('.feather'):
        return pd.read_feather(_path,**kwargs)
    elif _path.lower().endswith('.pkl'):
        return pd.read_pickle(_path,**kwargs)
    else:
        raise ValueError('File in wrong file format')


def gen_dataloader(_train_path,_test_path,batch_size,
                   preprocess_inputs = False,
                   tokenizer_type='bert-base-uncased',input_len=128,**kwargs):
    """
    Helper function that takes either just the train data path or both
    train and test data an outputs the appropriate dataloader instance

    kwargs are:
    for preprocessing:
    sample_size=None,
    weak_supervision=True
    max_len = 128
    filter_bad_rows = True
    tokenizer = DFAULT_TOKENIIZER
    
    For dataloaders:
    val_sample_dataloader=True
    pin_memory = False
    num_workers = 0
    """
    
    if 'bert' in tokenizer_type.lower():
        tokenizer = BertTokenizer.from_pretrained(tokenizer_type)
    elif 'xlnet' in tokenizer_type.lower():
        tokenizer = XLNetTokenizer.from_pretrained(tokenizer_type)
    else:
        raise NotImplementedError('model {} is not implemented'.format(tokenizer_type))
    
    train_dataset = read_data_to_dataframe(_train_path)
    if preprocess_inputs:
        df_train = preprocess_model_inputs(train_dataset,tokenizer=tokenizer,output_len=input_len,**kwargs)
    else:
        df_train = train_dataset
    
    if _test_path:
        test_dataset = read_data_to_dataframe(_test_path)        
        if preprocess_inputs:
            df_test = preprocess_model_inputs(test_dataset,tokenizer=tokenizer,**kwargs)
        else:
            df_test = test_dataset
        dl = TrainValDataloader(df_train,df_test,batch_size,kwargs)
        return dl
     
    
    dl = TrainValSplitDataloader(df_train,batch_size,kwargs)
    return dl    