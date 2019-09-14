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
    confusion = np.array([[0,0],[0,0]])

    for train_ind, test_ind in k_fold.split(dataprep.train_news):
        train_text = dataprep.train_news.iloc[train_ind]['Statement'] 
        train_y = dataprep.train_news.iloc[train_ind]['Label']
    
        test_text = dataprep.train_news.iloc[test_ind]['Statement']
        test_y = dataprep.train_news.iloc[test_ind]['Label']
        
        classifier.fit(train_text,train_y)
        predictions = classifier.predict(test_text)
        
        confusion += confusion_matrix(test_y,predictions)
        score = f1_score(test_y,predictions)
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

dataprep.test_news['Label'].shape

"""
Out of all the models fitted, we would take 2 best performing model. we would call them candidate models
from the confusion matrix, we can see that random forest and logistic regression are best performing 
in terms of precision and recall (take a look into false positive and true negative counts which appeares
to be low compared to rest of the models)
"""

#grid-search parameter optimization
#random forest classifier parameters
parameters = {'rf_tfidf__ngram_range': [(1, 1), (1, 2),(1,3),(1,4),(1,5)],
               'rf_tfidf__use_idf': (True, False),
               'rf_clf__max_depth': (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
}

gs_clf = GridSearchCV(random_forest_ngram, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(dataprep.train_news['Statement'][:10000],dataprep.train_news['Label'][:10000])

gs_clf.best_score_
gs_clf.best_params_
gs_clf.cv_results_

#logistic regression parameters
parameters = {'LogR_tfidf__ngram_range': [(1, 1), (1, 2),(1,3),(1,4),(1,5)],
               'LogR_tfidf__use_idf': (True, False),
               'LogR_tfidf__smooth_idf': (True, False)
}

gs_clf = GridSearchCV(logistic_regression_pipeline_ngram, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(dataprep.train_news['Statement'][:10000],dataprep.train_news['Label'][:10000])

gs_clf.best_score_
gs_clf.best_params_
gs_clf.cv_results_

#Linear SVM 
parameters = {'svm_tfidf__ngram_range': [(1, 1), (1, 2),(1,3),(1,4),(1,5)],
               'svm_tfidf__use_idf': (True, False),
               'svm_tfidf__smooth_idf': (True, False),
               'svm_clf__penalty': ('l1','l2'),
}

gs_clf = GridSearchCV(svm_pipeline_ngram, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(dataprep.train_news['Statement'][:10000],dataprep.train_news['Label'][:10000])

gs_clf.best_score_
gs_clf.best_params_
gs_clf.cv_results_

#by running above commands we can find the model with best performing parameters


#running both random forest and logistic regression models again with best parameter found with GridSearch method
random_forest_final = Pipeline([
        ('rf_tfidf',TfidfVectorizer(stop_words='english',ngram_range=(1,3),use_idf=True,smooth_idf=True)),
        ('rf_clf',RandomForestClassifier(n_estimators=300,n_jobs=3,max_depth=10))
        ])
    
random_forest_final.fit(dataprep.train_news['Statement'],dataprep.train_news['Label'])
predicted_rf_final = random_forest_final.predict(dataprep.test_news['Statement'])
np.mean(predicted_rf_final == dataprep.test_news['Label'])
print(metrics.classification_report(dataprep.test_news['Label'], predicted_rf_final))

logistic_regression_pipeline_final = Pipeline([
        #('LogRCV',countV_ngram),
        ('LogR_tfidf',TfidfVectorizer(stop_words='english',ngram_range=(1,5),use_idf=True,smooth_idf=False)),
        ('LogR_clf',LogisticRegression(penalty="l2",C=1))
        ])

logistic_regression_pipeline_final.fit(dataprep.train_news['Statement'],dataprep.train_news['Label'])
predicted_LogR_final = logistic_regression_pipeline_final.predict(dataprep.test_news['Statement'])
np.mean(predicted_LogR_final == dataprep.test_news['Label'])
#accuracy = 0.62
print(metrics.classification_report(dataprep.test_news['Label'], predicted_LogR_final))


"""
by running both random forest and logistic regression with GridSearch's best parameter estimation, we found that for random 
forest model with n-gram has better accuracty than with the parameter estimated. The logistic regression model with best parameter 
has almost similar performance as n-gram model so logistic regression will be out choice of model for prediction.
"""

#saving best model to the disk
model_file = 'final_model.sav'
pickle.dump(logistic_regression_pipeline_ngram,open(model_file,'wb'))


#Plotting learing curve
def plot_learing_curve(pipeline,title):
    size = 10000
    cv = KFold(size, shuffle=True)
    
    X = dataprep.train_news["Statement"]
    y = dataprep.train_news["Label"]
    
    pl = pipeline
    pl.fit(X,y)
    
    train_sizes, train_scores, test_scores = learning_curve(pl, X, y, n_jobs=-1, cv=cv, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)
       
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
     
    plt.figure()
    plt.title(title)
    plt.legend(loc="best")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.gca().invert_yaxis()
    
    # box-like grid
    plt.grid()
    
    # plot the std deviation as a transparent range at each training set size
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    # plot the average training and test score lines at each training set size
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    # sizes the window for readability and displays the plot
    # shows error from 0 to 1.1
    plt.ylim(-.1,1.1)
    plt.show()


#below command will plot learing curves for each of the classifiers
plot_learing_curve(logistic_regression_pipeline_ngram,"Naive-bayes Classifier")
plot_learing_curve(naive_bayes_pipeline_ngram,"LogisticRegression Classifier")
plot_learing_curve(svm_pipeline_ngram,"SVM Classifier")
plot_learing_curve(sgd_pipeline_ngram,"SGD Classifier")
plot_learing_curve(random_forest_ngram,"RandomForest Classifier")

"""
by plotting the learning cureve for logistic regression, it can be seen that cross-validation score is stagnating throughout and it 
is unable to learn from data. Also we see that there are high errors that indicates model is simple and we may want to increase the
model complexity.
"""


#plotting Precision-Recall curve
def plot_PR_curve(classifier):
    
    precision, recall, thresholds = precision_recall_curve(dataprep.test_news['Label'], classifier)
    average_precision = average_precision_score(dataprep.test_news['Label'], classifier)
    
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Random Forest Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))
    
plot_PR_curve(predicted_LogR_ngram)
plot_PR_curve(predicted_rf_ngram)


"""
Now let's extract the most informative feature from ifidf vectorizer for all fo the classifiers and see of there are any common
words that we can identify i.e. are these most informative feature acorss the classifiers are same? we will create a function that 
will extract top 50 features.
"""

def show_most_informative_features(model, vect, clf, text=None, n=50):
    # Extract the vectorizer and the classifier from the pipeline
    vectorizer = model.named_steps[vect]
    classifier = model.named_steps[clf]

     # Check to make sure that we can perform this computation
    if not hasattr(classifier, 'coef_'):
        raise TypeError(
            "Cannot compute most informative features on {}.".format(
                classifier.__class__.__name__
            )
        )
            
    if text is not None:
        # Compute the coefficients for the text
        tvec = model.transform([text]).toarray()
    else:
        # Otherwise simply use the coefficients
        tvec = classifier.coef_

    # Zip the feature names with the coefs and sort
    coefs = sorted(
        zip(tvec[0], vectorizer.get_feature_names()),
        reverse=True
    )
    
    # Get the top n and bottom n coef, name pairs
    topn  = zip(coefs[:n], coefs[:-(n+1):-1])

    # Create the output string to return
    output = []

    # If text, add the predicted value to the output.
    if text is not None:
        output.append("\"{}\"".format(text))
        output.append(
            "Classified as: {}".format(model.predict([text]))
        )
        output.append("")

    # Create two columns with most negative and most positive features.
    for (cp, fnp), (cn, fnn) in topn:
        output.append(
            "{:0.4f}{: >15}    {:0.4f}{: >15}".format(
                cp, fnp, cn, fnn
            )
        )
    #return "\n".join(output)
    print(output)

show_most_informative_features(logistic_regression_pipeline_ngram,vect='LogR_tfidf',clf='LogR_clf')
show_most_informative_features(naive_bayes_pipeline_ngram,vect='naive_bayes_tfidf',clf='naive_bayes_clf')
show_most_informative_features(svm_pipeline_ngram,vect='svm_tfidf',clf='svm_clf')
show_most_informative_features(sgd_pipeline_ngram,vect='sgd_tfidf',clf='sgd_clf')
