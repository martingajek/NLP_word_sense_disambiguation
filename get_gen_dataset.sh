#!/bin/sh
echo "Downloading wordnet to nltk"
python -c "import nltk; nltk.download('wordnet');"
echo " "
echo "Downloading semcor"
cd $PWD/data/raw/
git clone https://github.com/rubenIzquierdo/wsd_corpora.git 
cd wsd_corpora
cp -r semcor3.0 ../
cp -r semeval2007_task17_allwords ../
echo " "
echo "Removing spurious folders"

cd ..
rm -rf wsd_corpora
cd ../../
echo " "
echo "Building Train dataset"
python  $PWD/source/data_preprocessing/en_semcor3_wordnet/generate_dataset.py \
        --fpath=$PWD/data/raw/semcor3.0 --savepath=./data/preprocessed/semcor_gloss.pkl
echo " "
echo "Building Test dataset"
python  $PWD/source/data_preprocessing/en_semcor3_wordnet/generate_dataset.py \
        --semcor=False --fpath=$PWD/data/raw/semeval2007_task17_allwords \
        --savepath=./data/preprocessed/senseval_gloss.pkl
