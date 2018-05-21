

from sklearn import svm
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import metrics

train_data=pd.read_csv('/users/lbhavine/ML/amazon_baby_train.csv').dropna()
test_data=pd.read_csv('/users/lbhavine/ML/amazon_baby_test.csv').dropna()




print("Train data size:",train_data.shape)
print("Test data size:",test_data.shape)

#preprocessing
def review_to_words( raw_review ):
   
         
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review) 
    
    words = letters_only.lower().split()                             
    
    stops = set(stopwords.words("english"))                  
    
    
    meaningful_words = [w for w in words if not w in stops]   
    
    return( " ".join( meaningful_words ))   

train_reviews_clean = [review_to_words(rev) for rev in train_data['review']]
test_reviews_clean= [review_to_words(rev) for rev in test_data['review']]
print("Finished preprocessing of data".center(50,'-'))

vec=TfidfVectorizer(max_features=500,min_df=4)
train_input=vec.fit_transform(train_reviews_clean).todense()
train_output=train_data['rating']


test_input=vec.transform(test_reviews_clean).todense()
test_output=test_data['rating']

print(train_input.shape)
print(test_input.shape)

clf = svm.SVC(C=200,kernel="rbf")

clf.fit(train_input, train_output)
predicted_output=clf.predict(test_input)
print(accuracy_score(test_output, predicted_output))

