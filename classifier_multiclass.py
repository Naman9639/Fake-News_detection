import dataprep
import featureselection
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

#string to test
doc_new = ['obama is running for president in 2016']

#the feature selection has been done in featureselection.py module. here we will create models using those features for prediction

#first we will use bag of words techniques

#building classifier using naive bayes 
naive_bayes_pipeline = Pipeline([
        ('NBCV',featureselection.countV),
        ('naive_bayes_clf',MultinomialNB())])

naive_bayes_pipeline.fit(dataprep.train_news['Statement'],dataprep.train_news['Label'])
predicted_naive_bayes = naive_bayes_pipeline.predict(dataprep.test_news['Statement'])
np.mean(predicted_naive_bayes == dataprep.test_news['Label'])


#building classifier using logistic regression
logistic_regression_pipeline = Pipeline([
        ('LogRCV',featureselection.countV),
        ('LogR_clf',LogisticRegression())
        ])

logistic_regression_pipeline.fit(dataprep.train_news['Statement'],dataprep.train_news['Label'])
predicted_LogR = logistic_regression_pipeline.predict(dataprep.test_news['Statement'])
np.mean(predicted_LogR == dataprep.test_news['Label'])


#building Linear SVM classfier
svm_pipeline = Pipeline([
        ('svmCV',featureselection.countV),
        ('svm_clf',svm.LinearSVC())
        ])

svm_pipeline.fit(dataprep.train_news['Statement'],dataprep.train_news['Label'])
predicted_svm = svm_pipeline.predict(dataprep.test_news['Statement'])
np.mean(predicted_svm == dataprep.test_news['Label'])


#using SVM Stochastic Gradient Descent on hinge loss
sgd_pipeline = Pipeline([
        ('svm2CV',featureselection.countV),
        ('svm2_clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5))
        ])

sgd_pipeline.fit(dataprep.train_news['Statement'],dataprep.train_news['Label'])
predicted_sgd = sgd_pipeline.predict(dataprep.test_news['Statement'])
np.mean(predicted_sgd == dataprep.test_news['Label'])


#random forest
random_forest = Pipeline([
        ('rfCV',featureselection.countV),
        ('rf_clf',RandomForestClassifier(n_estimators=200,n_jobs=3))
        ])
    
random_forest.fit(dataprep.train_news['Statement'],dataprep.train_news['Label'])
predicted_rf = random_forest.predict(dataprep.test_news['Statement'])
np.mean(predicted_rf == dataprep.test_news['Label'])


#User defined functon for K-Fold cross validatoin
def build_confusion_matrix(classifier):
    n=len(dataprep.train_news)
    k_fold = KFold(n_splits=5)
    scores = []
    confusion = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])

    for train_ind, test_ind in k_fold.split(dataprep.train_news):
        train_text = dataprep.train_news.iloc[train_ind]['Statement'] 
        train_y = dataprep.train_news.iloc[train_ind]['Label']
    
        test_text = dataprep.train_news.iloc[test_ind]['Statement']
        test_y = dataprep.train_news.iloc[test_ind]['Label']
        
        classifier.fit(train_text,train_y)
        predictions = classifier.predict(test_text)
        
        confusion += confusion_matrix(test_y,predictions)
        score = f1_score(test_y,predictions, average=None)
        scores.append(score)
    
    return (print('Total statements classified:', len(dataprep.train_news)),
    print('Score:', sum(scores)/len(scores)),
    print('score length', len(scores)),
    print('Confusion matrix:'),
    print(confusion))
    
#K-fold cross validation for all classifiers
print('Bag of words confusion matrix')
print('Naive Bayes')
build_confusion_matrix(naive_bayes_pipeline)
print('Logistic Regression')
build_confusion_matrix(logistic_regression_pipeline)
print('SVM')
build_confusion_matrix(svm_pipeline)
print('SGD classifier')
build_confusion_matrix(sgd_pipeline)
print('Random Forest')
build_confusion_matrix(random_forest)

##Now using n-grams
#naive-bayes classifier
naive_bayes_pipeline_ngram = Pipeline([
        ('naive_bayes_tfidf',featureselection.tfidf_ngram),
        ('naive_bayes_clf',MultinomialNB())])

naive_bayes_pipeline_ngram.fit(dataprep.train_news['Statement'],dataprep.train_news['Label'])
predicted_naive_bayes_ngram = naive_bayes_pipeline_ngram.predict(dataprep.test_news['Statement'])
np.mean(predicted_naive_bayes_ngram == dataprep.test_news['Label'])


#logistic regression classifier
logistic_regression_pipeline_ngram = Pipeline([
        ('LogR_tfidf',featureselection.tfidf_ngram),
        ('LogR_clf',LogisticRegression(penalty="l2",C=1))
        ])

logistic_regression_pipeline_ngram.fit(dataprep.train_news['Statement'],dataprep.train_news['Label'])
predicted_LogR_ngram = logistic_regression_pipeline_ngram.predict(dataprep.test_news['Statement'])
np.mean(predicted_LogR_ngram == dataprep.test_news['Label'])


#linear SVM classifier
svm_pipeline_ngram = Pipeline([
        ('svm_tfidf',featureselection.tfidf_ngram),
        ('svm_clf',svm.LinearSVC())
        ])

svm_pipeline_ngram.fit(dataprep.train_news['Statement'],dataprep.train_news['Label'])
predicted_svm_ngram = svm_pipeline_ngram.predict(dataprep.test_news['Statement'])
np.mean(predicted_svm_ngram == dataprep.test_news['Label'])


#sgd classifier
sgd_pipeline_ngram = Pipeline([
         ('sgd_tfidf',featureselection.tfidf_ngram),
         ('sgd_clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5))
         ])

sgd_pipeline_ngram.fit(dataprep.train_news['Statement'],dataprep.train_news['Label'])
predicted_sgd_ngram = sgd_pipeline_ngram.predict(dataprep.test_news['Statement'])
np.mean(predicted_sgd_ngram == dataprep.test_news['Label'])


#random forest classifier
random_forest_ngram = Pipeline([
        ('rf_tfidf',featureselection.tfidf_ngram),
        ('rf_clf',RandomForestClassifier(n_estimators=300,n_jobs=3))
        ])
    
random_forest_ngram.fit(dataprep.train_news['Statement'],dataprep.train_news['Label'])
predicted_rf_ngram = random_forest_ngram.predict(dataprep.test_news['Statement'])
np.mean(predicted_rf_ngram == dataprep.test_news['Label'])


#K-fold cross validation for all classifiers
print('n-grams & tfidf confusion matrix and F1 scores')
print('Naive Bayes')
build_confusion_matrix(naive_bayes_pipeline_ngram)
print('Logistic Regression')
build_confusion_matrix(logistic_regression_pipeline_ngram)
print('SVM')
build_confusion_matrix(svm_pipeline_ngram)
print('SGD classifier')
build_confusion_matrix(sgd_pipeline_ngram)
print('Random Forest')
build_confusion_matrix(random_forest_ngram)


print(classification_report(dataprep.test_news['Label'], predicted_naive_bayes_ngram))
print(classification_report(dataprep.test_news['Label'], predicted_LogR_ngram))
print(classification_report(dataprep.test_news['Label'], predicted_svm_ngram))
print(classification_report(dataprep.test_news['Label'], predicted_sgd_ngram))
print(classification_report(dataprep.test_news['Label'], predicted_rf_ngram))
