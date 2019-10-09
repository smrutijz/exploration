import numpy as np
import pandas as pd
import nltk
from matplotlib import pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD

######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################

d = pd.read_excel(r'C:\Users\S5SBS7\Desktop\Python\Neural Network\InputData.xlsx')
d=d.fillna('')
sentences = d['Abstract']+d['Cause']+d['Comment']+d['EventName']+d['Type']

Y=[]
for _ in d['LCAT SenGroup']:
    if _ == '':
        Y.append(0)
    else:
        Y.append(1)

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
#SVD = TruncatedSVD(n_components=100).fit(d_features)
#d_features = pd.DataFrame(SVD.transform(d_features))

data = pd.concat([pd.DataFrame(Y,columns=['Label']),d_features],axis=1)
data = data.sample(frac=1).reset_index(drop=True)

sample_size = round(data.shape[0]*0.8)

train_data = np.array(data.iloc[:sample_size,1:])
train_label = np.array(data.iloc[:sample_size,0:1])

test_data = np.array(data.iloc[sample_size:,1:])
test_label = np.array(data.iloc[sample_size:,0:1])

######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################

M,n = train_data.shape
M_test,n_test = test_data.shape

num_itr = 1200
Alpha = 1
Lambda = 0.35
N=[n,10,1]
L = len(N)
min_odd = 0.6

W = []  #L Dim
B = []  #L Dim
J = []  #Num of Iteration Dim
J_test = []  #Num of Iteration Dim

A0 = train_data
norm = Normalizer().fit(A0)
A0 = norm.transform(A0)
A0_test = norm.transform(test_data)

Y = train_label
Y_test = test_label

#Making everthing L dimention. Easer to apply for loop for all layers
W.append([np.nan])
B.append([np.nan])

for l in range(1,L):
    #Multiply with 0.001 is essential. or else too big wt, will always end at extreme end of activation function.    
    W.append(np.random.rand(N[l]*N[l-1]).reshape(N[l],N[l-1])*0.1)
    B.append(np.random.rand(1*N[l]).reshape(1,N[l])*0.1)

##################################################

for i in range(0,num_itr):
    Z = []
    A = []
    Z_test = []
    A_test =[]
    dZ = []
    dA = []
    dW = []
    dB = []
    dG = []

    #Making everything L dimentional
    Z.append([np.nan])
    Z_test.append([np.nan])
    A.append(A0)
    A_test.append(A0_test)
    dG.append(np.nan)
    for j in range(0,L):
        dZ.append(np.nan)
        dA.append(np.nan)
        dW.append(np.nan)
        dB.append(np.nan)
    
    #Forward Propagation
    for l in range(1,L):
        Z.append(np.dot(A[l-1],W[l].T)+B[l])
        A.append(1/(1+np.exp(-Z[l])))

    #Cost
    W2_temp = 0
    for l in range(1,L):
        W2_temp+=(np.identity(W[l].shape[0])*np.dot(W[l],W[l].T)).sum(axis=1).sum(axis=0)
    
    J.append((-1*(Y*np.log(A[L-1])+(1-Y)*np.log(1-A[L-1]))).sum(axis=0)/M + Lambda*W2_temp/(2*M))
    
    for l in range(1,L):
        Z_test.append(np.dot(A_test[l-1],W[l].T)+B[l]) 
        A_test.append(1/(1+np.exp(-Z_test[l])))
        
    J_test.append((-1*(Y*np.log(A[L-1])+(1-Y)*np.log(1-A[L-1]))).sum(axis=0)/M_test)
    
    #Backward Propagation
    for j in range(1,L-1):
        dG.append((A[j]*(1-A[j])))
    dG.append(np.nan)

    for l in range(1,L):
        if l == 1:
            dZ[L-l] = A[L-l]-Y
        else:
            dZ[L-l] = dA[L-l]*dG[L-l]*Z[L-l]
        dW[L-l] = np.dot(dZ[L-l].T,A[L-l-1])/M
        dB[L-l] = dZ[L-l].sum(axis=0,keepdims=True)/M
        if L-l != 0:
            dA[L-l-1] = np.dot(dZ[L-l],W[L-l])

    #Update Weight & Bias      
    for l in range(1,L):
        W[l] = (1-Alpha*Lambda/M)*W[l] - Alpha*dW[l]
        B[l]-=Alpha*dB[l]

##################################################

plt.plot(J,color='blue')
plt.plot(J_test,color='red')
plt.xlabel('No of Iteration')
plt.ylabel('Cost Fuctionm')
plt.show()

##################################################


def accuracy(dt,label):
    NN_OP = []
    NN_OP.append(dt)
    for l in range(1,L):
        z_temp = np.dot(NN_OP[l-1],W[l].T)+B[l] 
        NN_OP.append(1/(1+np.exp(-z_temp)))

    OP = [(_[0]>min_odd)*1 for _ in NN_OP[L-1]]

    pred = pd.DataFrame(OP,columns=['Predicted'])
    actu = pd.DataFrame(label,columns=['Actual'])
    df_op = pd.concat([pred,actu],axis=1)

    ct = pd.crosstab(df_op['Predicted'],df_op['Actual'])
    t_acuracy = (ct.iloc[0,0]+ct.iloc[1,1])*100/ct.sum(axis=1).sum(axis=0)
    return t_acuracy, NN_OP[L-1]

print('Train accuracy',accuracy(train_data,train_label)[0])
print('Test accuracy',accuracy(test_data,test_label)[0])

######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
