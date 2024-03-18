import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB 

import seaborn as sns
import matplotlib.pyplot as plt

def train_model(alpha, tfidf_train, tfidf_test, y_train, y_test):
    """
    This function trains and tests a naive bayes classifiers.
    Inputs:
        - alpha: A smoothing parameter.
        - tfidf_train: the training features (A scipy sparse matrix of the features)
        - tfidf_test: the testing features (A scipy sparse matrix of the features)
        - y_train: the training labels (A pandas Serie)
        - y_test: the training labels (A pandas Serie)

    Outputs:
        - score: The classification accuracy.
    """

    # Instantiate the classifier: nb_classifier
    nb_classifier = MultinomialNB(alpha=alpha)

    # Fit to the training data
    nb_classifier.fit(tfidf_train, y_train)


    # Predict the labels: pred
    #pred = nb_classifier.predict(tfidf_test)

    # Compute accuracy: score
    #score = metrics.accuracy_score(y_test, pred)

    score = metrics.accuracy_score(y_train, nb_classifier.predict(tfidf_train))
    
    return score




def plot_alpha_score(alphas, scores, title):
    """
    This function plots accuracy at different alpha values.
    Inputs:
        - alphas: An ndarray of the alpha values.
        - scores: The corresponding accuracy values
        - title: The graph title
    """
    
    g = sns.lineplot(x=alphas, y=scores, marker="o", c="red", markersize=10, markerfacecolor='black')
    sns.despine(right=True, top=True)
    g.set_xlabel("Alpha", fontsize=14)
    g.set_ylabel("Accuracy", fontsize=14)
    g.set_title(title)
    plt.show()