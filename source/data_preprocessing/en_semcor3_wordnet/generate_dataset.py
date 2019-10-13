#!/usr/bin/env python
from tqdm import tqdm
import pandas as pd

import wordnet_gloss_search as wgs
import parse_semcor as pssc



def add_wordnet_gloss_byref(_semcordf,verbose=True):
    """ 
    Given a base semcor corpus dataframe generates gloss column for each word
    adds a corresponding to other glosses not relevant in each context
    """
    #if verbose: print('Adding wordnet glosses')
    tqdm.pandas(desc="Gloss preprocessing") 
    _semcordf['gloss']  = _semcordf.wn_index.progress_apply(wgs.wordnet_get_gloss_byref)
    #if verbose: print('Adding other wordnet glosses to semcor...',end="")
    tqdm.pandas(desc="Adding other glosses") 
    _semcordf['other_glosses']  = _semcordf.wn_index.progress_apply(wgs.wordnet_get_other_glosses_byref,{'select_name':True})
    # number of other glosses gives an idea of the ambiguity of each word   
    _semcordf['other_glossesnum'] = _semcordf['other_glosses'].apply(len)
    if verbose: print('Done!')
    return _semcordf

def add_wordnet_gloss(_semcordf,verbose=True):
    """ 
    Given a base semcor corpus dataframe generates gloss column for each word
    adds a corresponding to other glosses not relevant in each context
    """
    #if verbose: print('Adding wordnet glosses')
    _semcordf['idx'] = list(range(len(_semcordf))) #adding index for merging
    tqdm.pandas(desc="Gloss preprocessing") 
    _glosses = _semcordf[_semcordf.wn_sense_num != '0'].progress_apply(lambda _row: (*wgs.wordnet_gloss_helper(_row['lemma'],_row['wn_sense_num'])\
                                                                        ,_row['idx']),axis=1 )
    _df_glosses = pd.DataFrame(_glosses.values.tolist(),columns=['gloss','other_glosses','idx'])
    _merged = pd.merge(_semcordf,_df_glosses,on='idx',how='left').fillna('')
    # for now take only first gloss
    _merged['gloss'] = _merged.gloss.apply(lambda x: x[0] if x else '')
    # tag how many other glosses there are
    _merged['other_glossesnum'] = _merged.other_glosses.apply(lambda x: len(x))   
    if verbose: print('Done!')
    return _merged


def gen_sentence_context_pairs(_df):
    """
    Given a semcor corpus dataframe where there is one word per row (Dataframe just for one sentence)
    Concatenates the rows into a proper sentence
    finds ambiguous words (With more than one gloss) and generates 
    labeled sentence/context pairs for each ambiguous word.
    outputs a list of dictionaries.
    """
    
    concatenated_sentence = _df.text.str.cat(sep = ' ').replace(" '","'")
    basedct = {'context':concatenated_sentence,
               #'sent':_df.iloc[0].sent,
               'file':_df.iloc[0].file}

    semcor_sentences = []
    # Make sure there are other glosses and that the gloss column is not null
    for i,line in _df[(_df.other_glossesnum > 0) & (_df.gloss != 'WN Error') & (_df.gloss != '')].iterrows(): 

        # First append the proper context to dct with label True
        newbasedct = basedct.copy()
        newbasedct['target_word'] = line.text
        newbasedct['gloss'] = line.gloss
        newbasedct['is_proper_gloss'] = True
        semcor_sentences.append(newbasedct)
        # Then append all different contexes with False labels
        for other_glosses in line.other_glosses:
            newbasedct = basedct.copy()
            newbasedct['target_word'] = line.text
            newbasedct['gloss'] = other_glosses
            newbasedct['is_proper_gloss'] = False
            semcor_sentences.append(newbasedct)
                
    return semcor_sentences


def build_joint_dataset(_df):
    """
    Builds full dataset of labeled context/gloss pairs
    inputs are the full semcor dataframe (One word per row) with gloss and other glosses
    outputs full dataset dataframe
    """
    groupbyobj = _df.groupby(['sent','file'])
    full_dict_list = []
    for [sentnum,file],gp in tqdm(groupbyobj,total=len(groupbyobj)):
        full_dict_list.extend(gen_sentence_context_pairs(gp))
    cols = ['file','context','target_word','gloss','is_proper_gloss']
    return pd.DataFrame(full_dict_list)[cols]

def build_joint_semcor_gloss_corpus(_basepath,verbose=True,byref=False):
    """
    Given filepath to base folder of semcor3.0 corpus containing the xml files
    Parses corpus and generates joint context-gloss pairs from wordnet glosses 
    if byref is passed the dictionary lookup is done byy reference not by word   
    """
    
    semcor_corpus_df = pssc.build_semcor_corpus(_basepath,verbose=verbose)
    if byref:
        semcor_corpus_df = add_wordnet_gloss_byref(semcor_corpus_df,verbose=verbose)
    else:
        semcor_corpus_df = add_wordnet_gloss(semcor_corpus_df,verbose=verbose)
    if verbose: print('Processing adn labeling joint cintext-gloss pairs...',end="")
    final_corpus = build_joint_dataset(semcor_corpus_df)
    if verbose: print('Done!')
    return final_corpus

def build_joint_senseval_gloss_corpus(_basepath,verbose=True,**kwargs):
    """
    Given filepath to base folder of senseval 2007 task 17 corpus containing the xml files
    Parses corpus and generates joint context-gloss pairs from wordnet glosses    
    senseval cannot be built by word since there is no sense_id
    """
    
    senseval_corpus_df = pssc.build_semcor_corpus(_basepath,verbose=verbose)
    ## removes artefact in wordnet ref number
    senseval_corpus_df['wn_index'] = senseval_corpus_df['wn_index'].str.split('%').apply(lambda x:'%'.join(x[1:]))
    senseval_corpus_df = add_wordnet_gloss_byref(senseval_corpus_df,verbose=verbose)
    #else:
    #    senseval_corpus_df = add_wordnet_gloss(senseval_corpus_df,verbose=verbose)
    if verbose: print('Processing adn labeling joint cintext-gloss pairs...',end="")
    final_corpus = build_joint_dataset(senseval_corpus_df)
    if verbose: print('Done!')
    return final_corpus



if __name__=='__main__':
    from argparse import ArgumentParser
    import os

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = ArgumentParser(description='Process some integers.')
    parser.add_argument('--fpath',  type=str, default='./data/raw/',
                       help='File path to the semcor directory')    
    parser.add_argument('--savepath',  type=str, default='./data/preprocessed/semcor_gloss.feather',
                       help='save path to final semcor directory')
    parser.add_argument('--semcor',  type=str2bool, default=True,
                       help='save path to final semcor directory')                       
    args = parser.parse_args()
    print(args)
    
    if args.semcor == True:
        final_corpus = build_joint_semcor_gloss_corpus(args.fpath)
    else:
        final_corpus = build_joint_senseval_gloss_corpus(args.fpath)
    final_corpus.to_feather(args.savepath)


