# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 20:07:18 2019

@author: iwan
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
df=pd.read_csv('SMSSpamCollection', sep='\t', names=['Status','Massage'])
#a=len(df[df.Status=='spam'])
df.loc[df["Status"]=='ham',"Status"]=1
df.loc[df["Status"]=='spam',"Status"]=0
train_x=df["Massage"]
train_y= df["Status"]
#steaming dan lemmatisasi dulu
cv=TfidfVectorizer(min_df=1, stop_words='english')
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size =0.2, random_state=4)
x_traincv= cv.fit_transform(x_train)
b=x_traincv.toarray()
#print(cv.inverse_transform(b[0]))
#print(x_train.iloc[0])
mnb=MultinomialNB()
y_train=y_train.astype('int')
mnb.fit(x_traincv, y_train)
x_testcv=cv.transform(x_test)
pred=mnb.predict(x_testcv)
actual= np.array(y_test)
count =0
for i in range (len(pred)):
    if pred[i]==actual[i]:
        count=count+1
acc=100*count/len(pred)

