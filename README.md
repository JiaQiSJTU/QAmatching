# QA matching

Paper: [Qi Jia, Mengxue Zhang, Shengyao Zhang, Kenny Q. Zhu:
Matching Questions and Answers in Dialogues from Online Forums. ECAI 2020: 2046-2053](http://ebooks.iospress.nl/publication/55121)


## Requirements
* tensorflow>=1.0
* numpy

## Data
Download the session-level and pair-level dataset from [Google Drive](https://drive.google.com/file/d/1W6lTjThARhk1a4i6m95ySX4M7HWsfeHy/view?usp=sharing) and unzip the file under ./datafile.

The files includes:
* The annotated dialogues which are split into train/dev/test sets by 7:1:2
    * train-700.json
    * dev-100.json
    * test-200.json
* The reconstructed Q-NQ pairs used for our proposed model:
    * train-full.json
    * dev-full.json
    * test-full.json
* Pretrained word embeddings:
    * word_emb_reduce.txt
    * word2idx-new.json
    

## Getting Started

Here is the breakdown of the commands :

* Train the model with
```
python train.py
```

* Evaluate and interact with the model with (config.py batch_size=1)
```
python evaluate.py
```

* Get the session predictions
```
python session_predictions.py
python outputPredict_True_role_sen.py
```






