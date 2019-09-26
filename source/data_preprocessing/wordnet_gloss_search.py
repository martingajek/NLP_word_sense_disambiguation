from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError


##################################################################
#	 Methods to search wordnet corpus from NLTK toolkit
#    for gloss
##################################################################


def wordnet_get_gloss(_ref):
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
    
def get_other_senses(_ref,select_name=False):
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
    
def wordnet_get_other_glosses(_ref,select_name=False):  
    """given a wordnet reference number returns list of homonyms glosses"""
    
    if not _ref: # if ref is empty
        return ''

    other_senses = get_other_senses(_ref,select_name=select_name)
    #ipdb.set_trace()
    if not isinstance(other_senses,list): # if error
        return [other_senses]
    definition_list = [syn.definition() for syn in get_other_senses(_ref,select_name=select_name)]
    return definition_list