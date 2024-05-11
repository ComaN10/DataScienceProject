import pandas as pd
import numpy as np

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report #, confusion_matrix
from pycm import *
from sklearn.linear_model import LogisticRegression
# Ensemble bagging
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
# Ensemble boosting
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier


Complete_data = pd.read_csv("Complete_treated_dataset")
data_targets_y = Complete_data["fetal_health"].astype(int)

Complete_data_X = Complete_data.drop(["fetal_health"],axis=1)

Complete_data_X_train, Complete_data_X_test,  Complete_data_y_train, Complete_data_y_test \
    = train_test_split(Complete_data_X, data_targets_y, test_size=0.2, random_state=50)

data_reduction_pca = np.load('Complete_treated_reduction_pca.npy')
data_reduction_pca_X_train, data_reduction_pca_X_test, data_reduction_pca_y_train, data_reduction_pca_y_test \
    = train_test_split(data_reduction_pca, data_targets_y, test_size=0.2, random_state=50)

Complete_reduction_mmr = np.load('Complete_treated_reduction_mmr.npy')
Complete_reduction_mmr_X_train, Complete_reduction_mmr_X_test, Complete_reduction_mmr_y_train, Complete_reduction_mmr_y_test \
    = train_test_split(data_reduction_pca, data_targets_y, test_size=0.2, random_state=50)


def creat_model(data, modelname, model, train_X, train_y, test_X, test_y):

    print(f"Training with {modelname} for {data}")
    # train
    model.fit(train_X, train_y)

    # predictions
    y_pred = model.predict(test_X)

    # # Accuracy
    # accuracy = accuracy_score(y_pred, test_y)
    # #residuals = test_y - y_pred
    #
    conf_matrix = ConfusionMatrix(actual_vector=np.array(test_y), predict_vector=np.array(y_pred))
    #
    # # Compute accuracy
    # accuracy = accuracy_score(test_y, y_pred)
    #
    # # Compute classification report
    # report = classification_report(test_y, y_pred)

    print("results")
    # print(f"Accuracy: {accuracy}")
    # print(f"report {report}")
    print(f"Confusion {conf_matrix}")
    # print(f"cv_scores std: {residuals.std()}")
    # print(f"cv_scores mean: {residuals.mean()}")
    # print(f"cv_scores max: {y_pred.max()}")
    # print(f"cv_scores min: {y_pred.min()}")
    # print(f"cv_scores predictStd: {y_pred.std()}")




def create_models(model, model_name):

    creat_model("Complete_data", model_name, model, Complete_data_X_train, Complete_data_y_train,
                   Complete_data_X_test, Complete_data_y_test)
    creat_model("data_reduction_pca", model_name, model, data_reduction_pca_X_train,
                   data_reduction_pca_y_train, data_reduction_pca_X_test, data_reduction_pca_y_test)
    creat_model("Complete_reduction_mmr", model_name, model, Complete_data_X_train, Complete_data_y_train,
                   Complete_data_X_test, Complete_data_y_test)


GaussianNB = GaussianNB()
create_models(GaussianNB, "GaussianNB")

Dtree = DecisionTreeClassifier()
create_models(Dtree, "DecisionTreeClassifier")

Lregression = LogisticRegression()
create_models(Lregression, "LogisticRegression")


# Ensemble models
ETC = ExtraTreesClassifier()
create_models(ETC, "ExtraTreesClassifier")
baggingC = BaggingClassifier()
create_models(baggingC, "BaggingClassifier")
RForesC = RandomForestClassifier()
create_models(RForesC, "RandomForestClassifier")

# Ensemble model boosting
GBC = GradientBoostingClassifier()
create_models(GBC, "GradientBoostingClassifier")
Ada = AdaBoostClassifier()
create_models(Ada, "AdaBoostClassifier")
Hist = HistGradientBoostingClassifier()
create_models(Hist, "HistGradientBoostingClassifier")

#pycm
