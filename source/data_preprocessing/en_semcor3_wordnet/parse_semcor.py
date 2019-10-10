import os,glob,pathlib
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm

##################################################################
#	 Methods to preprocess semcor corpus from its underlying xml 
#    files into pandas dataframes
#    
##################################################################


def xml_parse_semcor(_fpath):
    """
    given semcor xml file path parses file and returns dataframe of words and references 
    to wordnet dictionary    
    """
    
    sctree = ET.parse(_fpath)

    # Iterates over list of words in files    
    dct_list1 = []
    for node in sctree.iter('wf'):
        attributes = node.attrib
        attributes['text'] = node.text
        dct_list1.append(attributes)

    # Iterates over terms to find senses and corresponding sense references
    dct_list2 = []
    for term in sctree.iter('term'):
        lemma = term.attrib.get('lemma')
        wordid = term.find('span/target').attrib.get('id')

        wnsn = '0'
        senseid=''
        if term.findall('externalReferences/externalRef'):
            wnsn = term.findall('externalReferences/externalRef')[0].attrib.get('reference')
            senseid = term.findall('externalReferences/externalRef')[1].attrib.get('reference')
        dct_list2.append({'id':wordid,'lemma':lemma,'wn_sense_num':wnsn,'lexical_key':senseid})

    word_df = pd.DataFrame(dct_list1)
    sense_ref_df = pd.DataFrame(dct_list2)   
    
    return pd.merge(word_df,sense_ref_df,on='id')


def gen_semcor_file_list(_basepath,ext='*.naf'):
    
    """
    given base path _basepath
    Builds a dataframe with all paths of files with ext extensions
    in subfolders
    """
    
    file_list = []
    fla = glob.glob(os.path.join(_basepath,ext))
    flb = glob.glob(os.path.join(_basepath,'*',ext))
    flc = glob.glob(os.path.join(_basepath,'**',ext))
    files = set(fla+flb+flc)
    
    for fileref in files: #search recursively for files
        parent_folder_name = pathlib.Path(fileref).parent.name
        file_name = pathlib.Path(fileref).name.split('.')[0]
        
        file_list.append( {'file_path':fileref,
                           'parent_folder':parent_folder_name,
                           'file_name':file_name})

    return pd.DataFrame(file_list)


def parse_semcor_corpus(_basepath,filter_validation = False):
    
    """
    Given a base directory, finds all '*.naf' files and
    generates a large pandas dataframe that includes all the corpus files
    
    """

   # generate dataframe with references to all files
    _fpath_df = gen_semcor_file_list(_basepath)
    
    # filter to remove validation files
    filtered_file_df = _fpath_df
    if filter_validation:
         filtered_file_df = _fpath_df[_fpath_df.parent_folder != 'brownv']
    
    _dflist = []
    for i,file_entry in tqdm(filtered_file_df.iterrows(), total=filtered_file_df.shape[0]):
        _parsed_file_df = xml_parse_semcor(file_entry.file_path)
        _parsed_file_df['file'] = file_entry.file_name
        _dflist.append(_parsed_file_df)

    return pd.concat(_dflist)

def build_semcor_corpus(_basepath,verbose=True,**kwargs):

	"""
    Parses and preprocesses semcor corpus
    
    """
	if verbose: print('Parsing corpus')
	base_corpus = parse_semcor_corpus(_basepath,**kwargs)

	# Build wordnet ref key using wordnet lemma
	if verbose: print('Preprocessing indexes...',end="")
	base_corpus['wn_index'] = base_corpus['lemma']+'%'+base_corpus['lexical_key']

	base_corpus.loc[base_corpus.lexical_key == '','wn_index'] = ''
	base_corpus.drop('lexical_key',axis=1,inplace=True)
	if verbose: print('Done!')
	return base_corpus


