import os
import numpy as np
import pickle
import pandas as pd
import nltk
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize

path = os.path.join(r'C:\\','Desktop','WikipediaToolKit','NLP')
os.chdir(path)

open_file = open("Training Data.pickle", "rb")
d = pickle.load(open_file)
open_file.close()

d=d.fillna('')

sentences = d['Description']+d['Comment']

y=[]
for _ in d['LCAT SenGroup']:
    if _ == '':
        y.append(0)
    else:
        y.append(1)

def token(text):
    word_token = nltk.tokenize.word_tokenize(text)
    return word_token

word_bag = []

for sentence in sentences:
    word_token = token(sentence)
    word_bag.extend(word_token)

word_bag = list(set(word_bag))

features_set=[]
for sentence in sentences:
    word_token = token(sentence)
    word_token = set(word_token)
    features={}
    for word in word_bag:
        features[word] = (word in word_token)*1
    features_set.append(features)

d_features = pd.DataFrame(features_set)

##################################################
m,n = d_features.shape

x = np.array(d_features)
y = np.array(y).reshape(m,1)
x = normalize(x)


w = np.random.rand(n).reshape(1,n)*0.001
b = np.random.rand(m).reshape(m,1)*0.001
alpha = [0.1]


cf = []
for itr2 in alpha: 
    j = []
    #gradient descent
    for itr1 in range(1,500):
        #forward propagation
        z = np.dot(x,w.transpose())+b
        a = 1/(1+np.exp(-z))
        #cost
        j.append(-1*(y*np.log(a)+(1-y)*np.log(1-a)))
        #backward propagation
        dw = (((a-y)*x).sum(axis=0)).reshape(1,n)
        db = (a-y).reshape(m,1)
        w-=alpha[0]*dw
        b-=alpha[0]*db
    cost = (np.array(j).sum(axis=1))/m
    cf.append(cost)

##################################################
plt.plot(cf[0]); plt.xlabel('No of Iteration'); plt.ylabel('Cost Fuctionm'); plt.show()
