import pandas as pd
import numpy as np
import csv
from sklearn import tree
import spacy
import en_core_web_sm

nlp = en_core_web_sm.load() 

#from sklearn.cross_validation import train_test_split

train_data = pd.read_csv('../data/train.csv')
test_X = pd.read_csv('../data/test.csv')
sub_data =pd.read_csv('../data/gender_submission.csv')



def get_similarity(term1, term2):
    tokens = nlp(term1 + " " + term2)

    print(tokens[0].text, "|",tokens[1].text, tokens[0].similarity(tokens[1]))

    return tokens[0].similarity(tokens[1])


train_Y = train_data['Survived']
train_data.drop(['Survived','Name','Ticket','Cabin','Embarked'], axis=1, inplace=True)
test_X.drop(['Name','Ticket','Cabin','Embarked'], axis=1, inplace=True)
train_X = train_data

for j in range(len(train_X['Pclass'])):
    if train_X['Sex'][j] == 'male' :
        train_X['Sex'][j] = 0
    elif train_X['Sex'][j] == 'female' :
        train_X['Sex'][j] = 1


for j in range(len(test_X['Pclass'])):
    if test_X['Sex'][j] == 'male':
        test_X['Sex'][j] = 0
    elif test_X['Sex'][j] == 'female':
        test_X['Sex'][j] = 1

train_X.fillna(0,inplace=True)
test_X.fillna(0,inplace=True)

clf = tree.DecisionTreeClassifier()
model = clf.fit(train_X,train_Y)

predict = model.predict(test_X)


count=0
for i in range(len(predict)):
    if predict[i]==sub_data['Survived'][i]:
        count+=1
        
print(count/len(predict))
pd.DataFrame(predict).to_csv('.predict.csv',index =False)
