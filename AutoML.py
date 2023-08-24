# -*- coding: utf-8 -*-
import pandas as pd
#%% Creating automatic machine learning program (classification)

'''
For this dataset we want to predict the wine quality(good=1, notgood=0) based on multiple Xs.
We will write a program which will try the differernt model for us at onece and store the performance for us.
We can then base on the performance of model to do model selection
'''

df = pd.read_csv('winequality-red.csv')
df = df.iloc[:,1:]

#%% Split to train, test, validation
from sklearn.model_selection import train_test_split
y = df['quality']
x = df.drop(['quality'],axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

#%%  Create sklearn models: Logistics Regression, KNN, Random Forest, SVM
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from time import time

model_svc = SVC()
model_logi = LogisticRegression()
model_knn = KNeighborsClassifier()
model_rf = RandomForestClassifier()

models = [{'name':'KNN', 
           'model':model_knn, 
           'hyperpara':{'n_neighbors':[3,5,10,15]
                        }
           }, 
          {'name':'Logistic Reg', 
           'model':model_logi, 
           'hyperpara':{'penalty':['l1', 'l2'],
                        'C': [0.1,1,10]
                        }
           }, 
          {'name':'SVM', 
           'model':model_svc, 
           'hyperpara':{'C': [1, 10], 
                        'kernel': ['linear', 'rbf']
                        }
           }, 
          {'name':'Random Froest', 
           'model':model_rf, 
           'hyperpara':{'n_estimators': [50,100,150],
                        'max_depth': [50,100,150,None]
                        }
           }
          ]


#%% GridSearchCV on every model&hyperparameter combinations
'''
We will use pd.Series to store the result of each model and combine all series into a dataframe in the final
'''

columns = ['model name', 'hyperparameter', 'cv score', 'time used', 'model']
results = []

for model_info in models:
    t_0 = time()
    model_name = model_info['name']
    model = model_info['model']
    parameters = model_info['hyperpara']
    model_cv = GridSearchCV(model, 
                            parameters, 
                            n_jobs=-1,
                            scoring='accuracy')
    model_cv.fit(x_train, y_train)
    print(model_cv.cv_results_)
    best_para = model_cv.best_params_
    cv_score = model_cv.best_score_
    best_model = model_cv.best_estimator_
    time_used = time()-t_0
    results.append(pd.Series([model_name, best_para, cv_score, time_used, best_model],
                             index=columns))
    
#%% Combine the results into a dataframe
df_result = pd.DataFrame(results, columns=columns)

#%% Choose best model basing on cv score
best_result = df_result.sort_values('cv score',ascending=False).iloc[0,:]
best_model = best_result.model

#%% Get test score
test_score = best_model.score(x_test, y_test)