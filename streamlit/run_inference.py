
#%%
import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import torch
import pandas as pd
from pytorch_transformers import BertTokenizer, BertModel
from keras.preprocessing.sequence import pad_sequences
import sys
from nltk.corpus import wordnet as wn
sys.path.append('../source/')
sys.path.append('../source/models')
sys.path.append('../source/dataloaders/')

from models import bert


import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


#from argparse import ArgumentParser
import sys

#TOKEN_LAYER = 'sent-cls-ws'
TOKEN_LAYER = 'token-cls'
OUTPUT_LEN = 128
WEAK_SUPERVISION = False


#%%
#HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
#HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
HTML_WRAPPER = """<mark class="blue">{}</mark>"""

def check_model_file():
    import urllib3
    import os
    import shutil
    url = 'https://drive.google.com/file/d/1z47HDDC4BLPlPBUHOC7YoEizOcDxSW59/view?usp=sharing'
    model_path = '../data/model_checkpoints/ModelWSD_ZTDOPC2AW5_ModelWSD_4.pth'
    
    if not os.path.exists(model_path):
        st.write('Downloading model')
        http = urllib3.PoolManager()

        with http.request('GET', url, preload_content=False) as r, open(model_path, 'wb') as out_file:       
            shutil.copyfileobj(r, out_file)
      
    return


def format_sentence_BERT(_sent,_gloss,_target,weak_supervision=False):
    """ Given dataframe input row, formats input sentence for tokenization:
    appends context to each sentence (repeats target word if weak_supervision)
    and appends [CLS] and [SEP] tags.   
    """
    if not weak_supervision:
        return '[CLS] '+_sent+' [SEP] '+_gloss+' [SEP]'
    return '[CLS] '+_sent+' [SEP] '+_target+': '+_gloss+' [SEP]'

def get_index_of_sep(_tokenized_sentence):
        """
        Get index of sep token and generate sentence index array
        """ 
        _index_sep_tokens = [i for i,word  in enumerate(_tokenized_sentence) \
                           if word == '[SEP]']
        _sentence_indexes = np.array([0]*(_index_sep_tokens[0]+1)\
                                     +[1]*(_index_sep_tokens[1]-_index_sep_tokens[0]))
        return _sentence_indexes
    
def find_index_of_target_token(_tokenized_sent,_tokenized_word):
    """
    looks for index of target token in the corresponding tokenized sentence
    
    """
    return [i for i,word  in \
           enumerate(_tokenized_sent) \
           if word == _tokenized_word.lower()]



def get_stopwords():
    return set(stopwords.words('english'))


@st.cache
def get_tokenizer():
    _tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return _tokenizer


@st.cache
def tokenize_input_sentence(input_text):
    TOKENIZER = get_tokenizer()
    return TOKENIZER.tokenize(input_text)

@st.cache
def tokenize_input_sentence(input_text):
    #stop_words = get_stopwords()
    TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized = TOKENIZER.tokenize(input_text)
    #return [token for token in tokenized if token not in stop_words]
    return tokenized


def process_data(_context,_target_word,_definitions,verbose=False,weak_supervision=WEAK_SUPERVISION):
    tokenizer = get_tokenizer()
    sentence_gloss = []
    #tokenized_sentence_gloss = []
    tokenized_target_word = tokenizer.tokenize(_target_word)[0]
    sentence_word_embeddings = []
    sentence_embeddings = []
    index_target_tokens = []
    _out_definitions = []
    
    bar = st.progress(0)
    for i,definition in enumerate(_definitions):
        bar.progress((i + 1)/len(_definitions))
        formatted_sentence = format_sentence_BERT(_context,definition,target_word,weak_supervision=weak_supervision)
        tokenized_sentence = tokenize_input_sentence(formatted_sentence)
        sentence_word_embedding = tokenizer.convert_tokens_to_ids(tokenized_sentence)
        sentence_embedding = get_index_of_sep(tokenized_sentence)
        index_target_token = find_index_of_target_token(tokenized_sentence,tokenized_target_word)
        if index_target_token:
            if verbose: print(formatted_sentence)
            #tokenized_sentence_gloss.append(tokenizer.tokenize(formatted_sentence))
            sentence_word_embeddings.append(np.array(sentence_word_embedding))
            sentence_embeddings.append(np.array(sentence_embedding))
            index_target_tokens.append(index_target_token[0])
            _out_definitions.append(definition)

    padded_words = pad_sequences(sentence_word_embeddings,maxlen=OUTPUT_LEN,dtype="long",padding = "post", truncating = "post")
    padded_sentences = pad_sequences(sentence_embeddings,maxlen=OUTPUT_LEN, dtype="long",padding = "post", truncating = "post",value=1)
    index_target_tokens = np.array(index_target_tokens)

    _tokens_tensor =torch.tensor(padded_words)
    _sentence_tensor = torch.tensor(padded_sentences)
    _target_token_ids = torch.tensor(index_target_tokens)
    b_size = _target_token_ids.shape[0]
    return _tokens_tensor, _sentence_tensor, _target_token_ids, _out_definitions

@st.cache
def get_definitions(_target_word):
    definitions = [syn.definition() for syn in wn.synsets(_target_word)]
    return definitions

@st.cache
def get_model():
    print('logging model again')
    #_model_path = '../data/model_checkpoints/run5/bertWSD_bertWSD_5.pth'
    _model_path = '../data/model_checkpoints/ModelWSD_ZTDOPC2AW5_ModelWSD_4.pth'
    model = bert.BertForWSD(output_logits=True,token_layer=TOKEN_LAYER)
    model.load_state_dict(torch.load(_model_path,map_location=torch.device('cpu')))
    ev = model.eval()
    return ev

@st.cache
def model_output(tokens_tensor, sentence_tensor, target_token_ids,_model):
    with torch.no_grad():
        _out = _model.forward(tokens_tensor, sentence_tensor, target_token_ids.unsqueeze(dim=1))
        _sm = torch.nn.Softmax(dim=1)
        _select = torch.argmax(_sm(_out),dim=1)
        return _select, _sm(_out)

@st.cache(suppress_st_warning=True)
def process_predict(_context,_target_word,verbose=False):
    _model = get_model()
    _definitions = get_definitions(target_word)
    _tokens_tensor, _sentence_tensor, _target_token_ids, _out_definitions = process_data(_context,
                                                                                        _target_word,
                                                                                        _definitions,
                                                                                        verbose=verbose)
    _out,_sm = model_output(_tokens_tensor, _sentence_tensor, _target_token_ids, _model)
    return _out.tolist(), _sm,_definitions

def rank_definitions(_softmax,_definitions):
    _out = torch.argmax(_softmax,dim=1).numpy()
    _scores = _softmax.numpy()
    _selected_scores = np.zeros_like(_out).astype(np.float32)    
    _selected_scores[np.where(_out == 1)] = _scores[_out == 1][:,1]
    _selected_scores[np.where(_out == 0)] = -_scores[_out == 0][:,0]
    _sorted_indices = np.argsort(_selected_scores)
    _ordered_dictionary = [_definitions[i] for i in _sorted_indices]
    return _out[_sorted_indices], _ordered_dictionary



    



    

#%%
#check_model_file()
st.sidebar.title('__Sense__ Finder')

filepath = sys.argv[0]
TOKENIZER = get_tokenizer()
target_word = st.sidebar.selectbox('Select Model Language',['English','Spanish'])
context = st.sidebar.text_input('Enter Sentence', '')
en_stopwords = set(stopwords.words('english'))

tokens = [""]+[word for word in word_tokenize(context) if word.lower() not in en_stopwords and word not in ',.']
target_word = st.sidebar.selectbox('Select Target word',tokens)



if target_word:
    
    st.title('Definitions of {}:'.format(target_word))
    out,sm,definitions = process_predict(context,target_word,verbose=False)
    
    #if st.sidebar.button('Rank'):
    #    out,definitions = rank_definitions(sm,definitions)

    for i,(out,definition) in enumerate(zip(out,definitions)):
        if out == 1:
            outstr = '{} {} ** {} **'.format(i,u'\u2713',definition.capitalize(),unsafe_allow_html=True)
            #st.markdown(HTML_WRAPPER.format(definition))
            st.write(HTML_WRAPPER.format(definition), unsafe_allow_html=True)
        else:
            st.write('{}'.format(i),' ',definition.capitalize())


    #st.write(sm.numpy())

        
    







#%%
