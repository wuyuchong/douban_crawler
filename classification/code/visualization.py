#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gensim
import itertools
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from pprint import pprint
import pickle
import re 
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import jieba
import matplotlib.pyplot as plt 
import pandas as pd
import nltk


# In[26]:


dat = pd.read_csv('./text/text.csv')
print(dat.query('rating == "bad"'))


# In[13]:


dat['text'] = dat.text.apply(lambda x: ",".join(jieba.cut(x)))
tweets = [t.split(',') for t in dat.text]
f = open('./code/ChineseStopWords.txt')
stopwords = f.read().splitlines()
f.close()
stopwords += ['....', '.....', 'end']
for i in range(len(tweets)):
    tweets[i] = [w for w in tweets[i] if w not in stopwords and len(w)>2]
text = dat.text.values.tolist()
words_list = list(itertools.chain(*tweets))


# In[14]:


print(tweets[0:5])


# In[15]:


id2word = Dictionary(tweets)
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in tweets]
print(corpus[:1])
[[(id2word[i], freq) for i, freq in doc] for doc in corpus[:1]]


# In[16]:


# Build LDA model
lda_model = LdaModel(corpus=corpus,
                   id2word=id2word,
                   num_topics=3, 
                   random_state=0,
                   chunksize=100,
                   alpha='auto',
                   per_word_topics=True)

pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# In[17]:


pyLDAvis.enable_notebook()
p = gensimvis.prepare(lda_model, corpus, id2word)
p


# In[ ]:




