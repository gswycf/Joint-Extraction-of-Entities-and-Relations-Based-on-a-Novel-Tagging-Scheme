# Joint extraction of entities and relations-pytorch+CNNs+LSTM
This project is based on [Deep Active Learning for Named Entity Recognition](https://arxiv.org/abs/1707.05928)  
and [Joint Extraction of Entities and Relations Based on a Novel Tagging Scheme](https://arxiv.org/abs/1706.05075)
    

# Dataset


    NYT-CoType
    
# Requirements

    1)python==3.6           
    2)pytorch==1.0.0
    3)numpy==1.14.5
    4)pickle
    5)gensim

# file

    ./data: data file + word2vec
    ./data.py:proces data file and save as binary file
    ./nodel.py: main model
    ./predict.py: not used in this project
    ./stats.py:statistics the data,not import in this project
    ./utils.py:define classes to keep wordvec
    ./model.pt:the model file
    ./record.tcv:the train record
    ./train.py: train file
    ./conv_net:CNN model

# how to run:

    python word2vec.py
    python data
    python train.py
    
    
