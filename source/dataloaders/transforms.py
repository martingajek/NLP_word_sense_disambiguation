import torch
from torch.nn import functional as F
from dataloaders import data_format_utils as dfu

def pad_tensor(_tensor,_pad_len):
    if _tensor.shape[0] >= _pad_len:
        _padded_tensor = _tensor[:_pad_len]
    else: # pad with 0s and 1s
        _padded_tensor = F.pad(_tensor,(0,_pad_len-_tensor.shape[0]))
    return _padded_tensor


def gen_embeddings(_row,weak_supervision=False,tokenizer=dfu.DEF_TOKENIZER):
    sent = dfu.format_sentences_BERT(_row,weak_supervision=weak_supervision)
    tok_sent = tokenizer.tokenize(sent)
    target_word_tokens = tokenizer.tokenize(_row.target_word)
    idx = dfu.find_target_token(tok_sent,target_word_tokens[0])
    token_idx_sent = tokenizer.convert_tokens_to_ids(tok_sent)
    sentence_embed = dfu.gen_sentence_embeddings(tok_sent)
    return torch.tensor(token_idx_sent), \
           torch.tensor(sentence_embed, dtype=torch.int64), \
           torch.tensor(idx, dtype=torch.int64)

def transform(_row,pad_len,weak_supervision=False,tokenizer=dfu.DEF_TOKENIZER):
    """
    giver row of pandas dataframe processes each input row into embeddings 
    returns 3 torch tensors with token embeddigs, sentence embeddings and
    target token index in the token embeddings tensor.
    """
    token_idxs,sentence_embed,target_token_idx = gen_embeddings(_row,
                                                                weak_supervision=weak_supervision,
                                                                tokenizer=tokenizer)
    token_idxs = pad_tensor(token_idxs,pad_len)
    sentence_embed = pad_tensor(sentence_embed,pad_len)
    return token_idxs, sentence_embed, target_token_idx   
    