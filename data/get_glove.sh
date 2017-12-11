#/bin/bash

glove='http://nlp.stanford.edu/data/glove.840B.300d.zip'
zipfile='glove.840B.300d.zip'
prune='prune_glove.py'
txtfile='glove/glove.840B.300d.txt'

mkdir 'glove'
cd 'glove'

wget "$glove"
unzip "$zipfile"

cd ..
python "$prune" "$txtfile"
