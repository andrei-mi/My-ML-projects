import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV
import re

# Importing the training and test datasets
dataset = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')



#Feature engineering on Names column
def title(name):
    title_search = re.search('([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

def surname(name):
    surname_search = re.search('([A-Za-z]+),', name)
    if surname_search:
        return surname_search.group(1)
    return ""

dataset['Title'] = dataset['Name'].apply(title)
dataset['Surname'] = dataset['Name'].apply(surname)

dataset['Title'] = dataset['Title'].replace(['Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col','Countess', 'Capt', 'Jonkheer'], 'Rare')
dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
dataset['Title'] = dataset['Title'].replace(['Mme'], 'Mrs')

data_test['Title'] = data_test['Name'].apply(title)
data_test['Surname'] = data_test['Name'].apply(surname)

data_test['Title'] = data_test['Title'].replace(['Dona','Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col','Countess', 'Capt', 'Jonkheer'], 'Rare')
data_test['Title'] = data_test['Title'].replace(['Mlle', 'Ms'], 'Miss')
data_test['Title'] = data_test['Title'].replace(['Mme'], 'Mrs')



X_train = dataset.drop(['Name', 'Cabin', 'Ticket', 'Age', 'Fare'], axis = 1)
y_train = dataset['Survived']

X_test = data_test.drop(['Name', 'Cabin', 'Ticket', 'Age', 'Fare'], axis = 1)

#Getting the list of boys in the test data and predicting their survival to be 1 in case all their female and boys
#relatives in the training data have survived. in case there are no such relatives, we don't make any changes
master_test = X_test.loc[X_test['Title'] == 'Master']

y_pr = pd.Series(np.zeros(418) , index = range(892,1310))

l1 = []
for i in master_test['Surname']:
     if len(X_train.loc[(X_train['Surname'] == i) & (X_train['Title'].isin(['Mrs', 'Miss', 'Master']))]['Survived']) > 0  and sum(X_train.loc[(X_train['Surname'] == i) & (X_train['Title'].isin(['Mrs', 'Miss', 'Master']))]['Survived']) == len(X_train.loc[(X_train['Surname'] == i) & (X_train['Title'].isin(['Mrs', 'Miss', 'Master']))]['Survived']):
         print(i)
         print(master_test.loc[master_test['Surname'] == i]['PassengerId'])
         for j in range(len(master_test.loc[master_test['Surname'] == i].index)):
             l1.append(master_test.loc[master_test['Surname'] == i].index[j])

l1 = list(np.unique(np.sort(np.array(l1)))+892)


y_pr[l1] = 1

#Getting the PassengerId list of women in the test data and predicting their survival to be 0 in case all their female and boys
#relatives in the training data died. in case there are no such relatives, we don't make any changes
female_test = X_test.loc[X_test['Sex'] == 'female']

l2 = []
for i in female_test['Surname']:
     if len(X_train.loc[(X_train['Surname'] == i) & (X_train['Title'].isin(['Mrs', 'Miss', 'Master']))]['Survived']) > 0  and sum(X_train.loc[(X_train['Surname'] == i) & (X_train['Title'].isin(['Mrs', 'Miss', 'Master']))]['Survived']) == 0:
         print(i)
         print(female_test.loc[female_test['Surname'] == i]['PassengerId'])
         for j in range(len(female_test.loc[female_test['Surname'] == i].index)):
             l2.append(female_test.loc[female_test['Surname'] == i].index[j])

l2 = list(np.unique(np.sort(np.array(l2)))+892)

all_data = pd.concat([dataset.drop('Survived', axis = 1), data_test])

#Preparing training data
all_data['Embarked'].fillna('S', inplace = True)
all_data['Sex'] = all_data['Sex'].map({'female': 1, 'male': 0})
all_data['Age'] = all_data['Age'].fillna(all_data['Age'].mean())
all_data['Fare'] = all_data['Fare'].fillna(all_data['Fare'].mean())

#Feature engineering
bins = [0, 12, 18, 25, 60,all_data['Age'].max()]
levels = ['child', 'teen', 'young', 'adult', 'elderly']
all_data['Age Category'] = pd.cut(all_data.Age, bins, labels=levels)

fare_bins = [0, 10, 20, 50, all_data['Fare'].max()]
fare_levels = ['low', 'median', 'average', 'high']
all_data['Fare_Category'] = pd.cut(all_data.Fare, fare_bins, labels=fare_levels)
all_data['Fare_Category'] = all_data['Fare_Category'].fillna('low')


X_train = all_data.drop(['Name', 'Cabin', 'Ticket', 'Age', 'Fare', 'PassengerId', 'Surname', 'Title'], axis = 1).values[:891,:]
y_train = dataset['Survived'].values

X_test = all_data.drop(['Name', 'Cabin', 'Ticket', 'Age', 'Fare', 'PassengerId', 'Surname', 'Title'], axis = 1).values[891:, :]


df_train = pd.DataFrame(X_train)

#Dummy variables
labelencoder_X = LabelEncoder()
X_train[:,4] = labelencoder_X.fit_transform(X_train[:,4])
X_train[:,5] = labelencoder_X.fit_transform(X_train[:,5])
X_train[:,6] = labelencoder_X.fit_transform(X_train[:,6])
onehotencoder = OneHotEncoder(categorical_features = [4,5,6])
X_train = onehotencoder.fit_transform(X_train).toarray()

labelencoder_X = LabelEncoder()
X_test[:,4] = labelencoder_X.fit_transform(X_test[:,4])
X_test[:,5] = labelencoder_X.fit_transform(X_test[:,5])
X_test[:,6] = labelencoder_X.fit_transform(X_test[:,6])
onehotencoder = OneHotEncoder(categorical_features = [4,5,6])
X_test = onehotencoder.fit_transform(X_test).toarray()

#SVC
#Fitting Kernel SVC to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predict the Test set results
y_pred = pd.Series(classifier.predict(X_test))

#Output results
results = pd.concat([data_test['PassengerId'], y_pred], axis = 1)
results.to_csv('Titanic_my_SVC_predictions.csv')


#Best score yet on kaggle: 0.808
#Parameter tuning
from sklearn.svm import SVC
model = SVC()
param_grid = {'kernel': ['rbf','linear'],
              'gamma' : [0.001, 0.01, 0.1, 1],
              'C': [1,10,50,100,200,500,1000]}
model_svc = GridSearchCV(model, param_grid = param_grid, cv = 5, scoring = "accuracy", n_jobs = 4, verbose = 1)
model_svc.fit(X_train, y_train)

#best estimate
print(model_svc.best_score_)

#best estimator
print(model_svc.best_estimator_)

classifier = model_svc.best_estimator_
classifier.fit(X_train, y_train)

#Predict the test set results
y_pred = pd.Series(classifier.predict(X_test), index = range(892,1310))

y_pred[l1] = 1
y_pred[l2] = 0

#Output results
results = np.vstack([data_test['PassengerId'].values, y_pred.values]).T
submission = pd.DataFrame(results, columns = ['PassengerId', 'Survived'])
submission.to_csv('Titanic_my_tuned_SVC-GM_predictions.csv' , index = False)
