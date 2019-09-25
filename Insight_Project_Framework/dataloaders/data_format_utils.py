import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from pytorch_transformers import BertTokenizer

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
        return '[CLS] '+_row['sent_full']+' [SEP] '+_row['context']+' [SEP]'
    return '[CLS] '+_row['sent_full']+' [SEP] '+_row['target_word']+': '+_row['context']+' [SEP]'



def tokenize_and_index(_df,max_len=MAX_LEN,tokenizer=DEF_TOKENIZER,weak_supervision=False):
    """
    Given corpus dataframe with one sentence per row as well as target word and definition
    preprocesses input sentence (adds start/sep tokens and appends context) then
    tokenizes both input and target word in the dataframe and converts each tokenized sentence 
    to input_id tensor. Generates padded input column for the input_id tensor (with trailing zeros)
    
    """
    
    _df['preproc_sent'] = _df.apply(format_sentences_BERT,axis=1,weak_supervision=weak_supervision)
    _df['tokenized_sent'] = _df.preproc_sent.apply(tokenizer.tokenize)
    _df['tokenized_target_word'] = _df.target_word.apply(lambda row: tokenizer.tokenize(row)[0])
    _df['input_ids'] = _df.tokenized_sent.apply(tokenizer.convert_tokens_to_ids)
    
    padded_input_ids = pad_sequences(_df['input_ids'], 
                                     maxlen=max_len, dtype="long",padding = "post", truncating = "post")
    _df['input_ids'] = np.split(padded_input_ids, _df.shape[0], axis=0)
    
    
def gen_sentence_indexes(_df,max_len=MAX_LEN):
    """
    given input dataframe with on tokenized sentence per row
    generates input sentence tensor and pads it to MAX_LEN with trailing 1's (2nd sentence)
    """
    
    def get_index_of_sep(_row):
        # Get index of sep token and generate sentence index array
        _index_sep_tokens = [i for i,word  in enumerate(_row['tokenized_sent']) \
                           if word == '[SEP]']
        _sentence_indexes = np.array([0]*(_index_sep_tokens[0]+1)\
                                     +[1]*(_index_sep_tokens[1]-_index_sep_tokens[0]))
        return _sentence_indexes
    
    _df['sent_indexes'] = _df.apply(get_index_of_sep,axis=1)
    padded_sent_idx = pad_sequences(_df['sent_indexes'],
                                               maxlen=MAX_LEN, dtype="long",
                                               padding = "post", truncating = "post",value=1)
    _df['sent_indexes'] = np.split(padded_sent_idx, _df.shape[0], axis=0)
    

def find_index_of_target_token(_df):
    """
    looks for index of target token in the corresponding tokenized sentence
    
    """
    find_token = lambda  _row: [i for i,word  in \
                         enumerate(_row['tokenized_sent']) \
                         if word == _row['tokenized_target_word'].lower()]
    _df['target_token_idx'] = _df.apply(find_token,axis=1)
    
    
    