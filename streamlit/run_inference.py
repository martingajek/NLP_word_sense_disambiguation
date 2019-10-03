
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

MODEL_PATH = '../data/model_checkpoints/run5/bertWSD_bertWSD_5.pth'

from models import bert
import ipdb
from nltk.corpus import stopwords


OUTPUT_LEN = 128

#%%

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


@st.cache
def process_data(_context,_target_word,_definitions,verbose=False):
    tokenizer = get_tokenizer()
    sentence_gloss = []
    #tokenized_sentence_gloss = []
    tokenized_target_word = tokenizer.tokenize(_target_word)[0]
    sentence_word_embeddings = []
    sentence_embeddings = []
    index_target_tokens = []
    _out_definitions = []
    for definition in _definitions:
        formatted_sentence = format_sentence_BERT(_context,definition,target_word,weak_supervision=True)
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
    model = bert.BertForWSD(output_logits=True,token_layer='sent-cls-ws')
    model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device('cpu')))
    ev = model.eval()
    return model

@st.cache
def model_output(tokens_tensor, sentence_tensor, target_token_ids,_model):
    with torch.no_grad():
        _out = model.forward(tokens_tensor, sentence_tensor, target_token_ids.unsqueeze(dim=1))
        _sm = torch.nn.Softmax(dim=1)
        _select = torch.argmax(_sm(_out),dim=1)
        return _select, _sm(_out)

    

#%%

TOKENIZER = get_tokenizer()
#model = bert.BertForWSD(output_logits=True,token_layer='sent-cls-ws')


model = get_model()

context = st.sidebar.text_input('Enter Sentence', 'default')
target_word = st.sidebar.multiselect('explanation2',tokenize_input_sentence(context))
if target_word:
    target_word = target_word[0]
    st.title('Definitions of {}:'.format(target_word))
    definitions = get_definitions(target_word)

    _tokens_tensor, _sentence_tensor, _target_token_ids, _out_definitions = process_data(context,target_word,definitions,verbose=False)
    out,sm = model_output(_tokens_tensor, _sentence_tensor, _target_token_ids,_out_definitions)
    
    outputs = out.tolist()

    for i,(_out,definition) in enumerate(zip(outputs,definitions)):
        if _out == 1:
            outstr = '{} {} ** {} **'.format(i,u'\u2713',definition.capitalize(),unsafe_allow_html=True)
            st.markdown(outstr)
        else:
            st.write('{}'.format(i),' ',definition.capitalize())

    #st.write(out)
    #st.write(sm)

    







#%%
