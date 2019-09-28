import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from pytorch_transformers import BertTokenizer
from tqdm import tqdm

################################################################################################
#                                                                                             #
# Helper functions to preprocess input corpus pandas dataframe containing one word            #
# per row into a usable format for word sense disambiguation model ingestion. The input       #
# dataframe needs 3 columns, namely ['sent_full','target_word','context'], where sent_full    #
# is the full sentence to be disambiguated, the target word is the word to be disambiguated   #
# and context is the gloss corresponding to the target word.                                  #
# The library tokenizes and indexes as well as pads each token. Generates a sentence index    #
# array for each sentence and finds the index of the target token in each tokenized sentence. #
#                                                                                             #
################################################################################################


MAX_LEN = 128
DEF_TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')


def format_sentences_BERT(_row,weak_supervision=False):
    """ Given dataframe input row, formats input sentence for tokenization:
    appends context to each sentence (repeats target word if weak_supervision)
    and appends [CLS] and [SEP] tags.   
    """
    if not weak_supervision:
        return '[CLS] '+_row.loc['sent']+' [SEP] '+_row.loc['gloss']+' [SEP]'
    return '[CLS] '+_row.loc['sent']+' [SEP] '+_row.loc['target_word']+': '+_row.loc['gloss']+' [SEP]'



def tokenize_and_index(_df,output_len=MAX_LEN,tokenizer=DEF_TOKENIZER,
                       weak_supervision=False,display_progress = True):
    """
    Given corpus dataframe with one sentence per row as well as target word and definition
    preprocesses input sentence (adds start/sep tokens and appends context) then
    tokenizes both input and target word in the dataframe and converts each tokenized sentence
    to input_id tensor. Generates padded input column for the input_id tensor (with trailing zeros)    
    """
   
   
    tqdm.pandas(desc="Sentence preprocessing")    
    _df.loc[:,'preproc_sent'] = _df.progress_apply(format_sentences_BERT,axis=1,weak_supervision=weak_supervision)
    tqdm.pandas(desc="Sentence Tokenization")
    _df.loc[:,'tokenized_sent'] = _df.preproc_sent.progress_apply(tokenizer.tokenize)
    tqdm.pandas(desc="Tokenizing target words")
    _df.loc[:,'tokenized_target_word'] = _df.target_word.progress_apply(lambda row: tokenizer.tokenize(row)[0])
    tqdm.pandas(desc="Converting tokens to embeddings")
    _df.loc[:,'input_ids'] = _df.tokenized_sent.progress_apply(tokenizer.convert_tokens_to_ids)
    
    padded_input_ids = pad_sequences(_df['input_ids'], 
                                    maxlen=output_len, dtype="long",padding = "post", truncating = "post")
    _df.loc[:,'input_ids'] = np.split(padded_input_ids, _df.shape[0], axis=0)
    
    
def gen_sentence_indexes(_df,output_len=MAX_LEN):
    """
    given input dataframe with on tokenized sentence per row
    generates input sentence tensor and pads it to MAX_LEN with trailing 1's (2nd sentence)
    """
    
    def get_index_of_sep(_row):
        """
        Get index of sep token and generate sentence index array
        """ 
        _index_sep_tokens = [i for i,word  in enumerate(_row['tokenized_sent']) \
                           if word == '[SEP]']
        _sentence_indexes = np.array([0]*(_index_sep_tokens[0]+1)\
                                     +[1]*(_index_sep_tokens[1]-_index_sep_tokens[0]))
        return _sentence_indexes
    
    tqdm.pandas(desc="Indexing sentences") 
    _df.loc[:,'sent_indexes'] = _df.progress_apply(get_index_of_sep,axis=1)
    padded_sent_idx = pad_sequences(_df['sent_indexes'],
                                               maxlen=MAX_LEN, dtype="long",
                                               padding = "post", truncating = "post",value=1)
    _df.loc[:,'sent_indexes'] = np.split(padded_sent_idx, _df.shape[0], axis=0)
    

def find_index_of_target_token(_df):
    """
    looks for index of target token in the corresponding tokenized sentence
    
    """
    find_token = lambda  _row: [i for i,word  in \
                         enumerate(_row['tokenized_sent']) \
                         if word == _row['tokenized_target_word'].lower()]
    tqdm.pandas(desc="Finding target token in sentence") 
    _df.loc[:,'target_token_idx'] = _df.progress_apply(find_token,axis=1)



def preprocess_model_inputs(_df,sample_size=100, filter_bad_rows=True,output_len=MAX_LEN,weak_supervision=False):
    """
    given preprocessed corpus dataframe tokenizes and creates the embeddings for
    input for the tranformer model. Furthermore it filters bad rows where the index
    of target word is larger than the size of each tokenized swquence.
    """
    
    _smpldf = _df
    if sample_size:
        _smpldf = _df.sample(sample_size)
    
    tokenize_and_index(_smpldf,output_len=output_len,weak_supervision=weak_supervision)
    gen_sentence_indexes(_smpldf,output_len=output_len)
    find_index_of_target_token(_smpldf)
    
    if filter_bad_rows: # rows where the target word index exceeds tensor size 
        _smpldf = _smpldf[_smpldf.target_token_idx.apply(lambda x: x[0] <  output_len)]

    
    return _smpldf
    
    
    