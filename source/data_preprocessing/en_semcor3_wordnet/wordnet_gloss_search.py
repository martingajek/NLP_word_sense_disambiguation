from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError


##################################################################
#	 Methods to search wordnet corpus from NLTK toolkit
#    for gloss
##################################################################


def wordnet_get_gloss_byref(_ref):
    """
    given a wordnet reference number returns term gloss
    _ref is the wordnet referenace number (e.g friday%1:28:00::)
    
    """
    if not _ref: # if ref is empty
        return ''
    try:
        return wn.synset_from_sense_key(_ref).definition()
    except (AttributeError,WordNetError,ValueError) as err:
        return 'WN Error'
    
def get_other_senses_byref(_ref,select_name=False):
    """
    given a wordnet reference number returns homonyms list in the form of
    other synsets
    _ref is the wordnet reference number (e.g friday%1:28:00::)  
    """

        
    try:
        ref_syn = wn.synset_from_sense_key(_ref)
        ref_word = ref_syn.name().split('.')[0]
        if not select_name:
            return [syn for syn in wn.synsets(ref_word) if syn != ref_syn]
        return [syn for syn in wn.synsets(ref_word) if (syn != ref_syn and ref_word in syn.name())]
    except (AttributeError,WordNetError,KeyError,ValueError) as err:
        return 'WN Error'   
    
def wordnet_get_other_glosses_byref(_ref,select_name=False):  
    """given a wordnet reference number returns list of homonyms glosses"""
    
    if not _ref: # if ref is empty
        return ''

    other_senses = get_other_senses_byref(_ref,select_name=select_name)
    if not isinstance(other_senses,list): # if error
        return [other_senses]
    definition_list = [syn.definition() for syn in get_other_senses_byref(_ref,select_name=select_name)]
    return definition_list

def wordnet_get_glosses(_word,_sense_id):
    """
    given a wordnet word (the lemma of the target word) number returns tuple of term glosses
    the first one corresponds to the _sense_id, the 2nd on is a list
    of all other glosses
    """
    _sense_id = int(_sense_id)
    if not _word: # if ref is empty
        return ''
    try:
        all_synsets = wn.synsets(_word)
        target_gloss = []
        other_glosses = []
        for syn in all_synsets:
            split = syn.name().split('.')
            wn_lemma = split[0]
            sense_num = int(split[-1])
            #if _word == wn_lemma:    
            if sense_num == _sense_id:
               target_gloss.append(syn.definition()) 
            else:
               other_glosses.append(syn.definition())                
        return target_gloss,other_glosses
    except (AttributeError,WordNetError,ValueError) as err:
        return 'WN Error',None
    
def wordnet_gloss_helper(_word,_sense_id):
    """takes target word and _Sense_id (str) and returns tuple of 
    glosses and other glosses
    """
    if not _word or not _sense_id:
        return '',''
    senseidlist = _sense_id.split(';')
    if len(senseidlist) == 1:
        return wordnet_get_glosses(_word,int(_sense_id))
    elif len(senseidlist) > 1:
        list_proper_glosses = []
        other_gloss_set = set()
        for senseid in senseidlist:
            gloss, other_glosses =  wordnet_get_glosses(_word,int(senseid))
            if gloss:
                list_proper_glosses.append(gloss)
                other_gloss_set.update(set(other_glosses))
        # if one of the flosses is bogus return only one
        if len(list_proper_glosses) == 1:
            return list_proper_glosses[0], other_gloss_set
        return list_proper_glosses, other_gloss_set
    else:
        return  'WN Error',[]   