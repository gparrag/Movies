#!/usr/bin/python

import tensorflow as tf
import keras
import numpy as np 
from sentence_transformers import SentenceTransformer
import sys
import os
import warnings
warnings.filterwarnings('ignore')
import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
warnings.filterwarnings('ignore')

cols = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
        'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
        'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']

def preprocess(df_text):
    df_text=df_text.lower()
    df_text=df_text.replace("'","")
    df_text=df_text.replace("“","")
    df_text=df_text.replace("”","")
    df_text=df_text.replace('—','')
    df_text=df_text.replace('’','')
    df_text=df_text.replace('‘','')
    df_text=re.sub(" \d+", " ",df_text)
    df_text=re.sub("\\W"," ",df_text)
    df_text=re.sub("/\b(\w)\b/","",df_text)
    df_text = re.sub('<[^>]*>', '', df_text)# remove none alphanumeric text
    df_text = re.sub('<.*?>+', '', df_text)
    df_text = re.sub(r"\s+[a-zA-Z]\s+", ' ', df_text)
    df_text = re.sub('[%s]' % re.escape(string.punctuation), '', df_text)
    tokenized_list = nltk.word_tokenize(df_text)
    stop_words=set(stopwords.words('english')) 
    cleaned_list=[i for i in tokenized_list if i not in stop_words]
    return ' '.join(cleaned_list)


def predict_genre(plot):
    #load model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    #embeded model
    model_embed = SentenceTransformer('bert-base-uncased')
    model_embed.max_seq_length = 110
    plot=preprocess(plot)
    plot_embed = model_embed.encode(plot)
    plot_embed = plot_embed.reshape(1,1,plot_embed.shape[0])
    genres= loaded_model.predict(plot_embed)
    genres=genres.reshape(1,24)
    
    genrespred=[]
    for i in range(genres.shape[1]):
        if genres[:,i] >= 0.5:
            genrespred.append(cols[i])
   
    return genrespred
    
if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please enter the movie plot')
        
    else:

        Plot = sys.argv[1]

        prediction = predict_genre(Plot)
    
        print('Movies Genres: ', prediction)  