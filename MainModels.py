import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, classification_report #, confusion_matrix
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
# Clustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

from KFoldCrossValidator import KFoldCrossValidator

from KNN import KNN
import joblib

# loading data
Complete_data = pd.read_csv("Complete_treated_dataset")

# separating labels
data_targets_y = Complete_data["fetal_health"].astype(int)
Complete_data_X = Complete_data.drop(["fetal_health"], axis=1)

# splitting in training and test
Complete_data_X_train, Complete_data_X_test,  Complete_data_y_train, Complete_data_y_test = train_test_split(Complete_data_X, data_targets_y, test_size=0.2, random_state=50)

def creat_model_with_cross_validation(name, model, x, y, CrossValidations=3):
    """
    Creates and fits a model with x, y data, trains and tests.
    :param name: fileName
    :param model: Model
    :param x: x_data pd
    :param y: label data pd
    :param CrossValidations: number of k-folds cross-validations
    :return: None
    """
    kFoldCrossValidator = KFoldCrossValidator(3)
    accuracy, sensitivity, specificity = kFoldCrossValidator.cross_validate(model, np.array(x), np.array(y))
    print("Accuracy:", accuracy[1])
    print("Sensitivity:", sensitivity[1])
    print("Specificity:", specificity[1])
    joblib.dump(model, "Models\\"+name)


def creat_model_without_cross_validation(name, model,
                                         X_Train=Complete_data_X_train,
                                         x_Test=Complete_data_X_test,
                                         Y_Train=Complete_data_y_train,
                                         y_Test=Complete_data_y_test):
    """
    Creates and fits a model with Complete_data_X_train, Complete_data_y_train data, trains and tests.
    :param name: fileName
    :param model: Model
    :return: None
    """
    model.fit(X_Train, Y_Train)
    y_pred = model.predict(x_Test)
    conf_matrix = ConfusionMatrix(actual_vector=np.array(y_Test), predict_vector=np.array(y_pred))
    print(conf_matrix)
    joblib.dump(model, "Models\\"+name)

# KNN
print("KNN Models:")
creat_model_with_cross_validation("KNN_Model",KNN(5),Complete_data_X,data_targets_y,3)

knn = KNN(5)
knn.fit(Complete_data_X_train.values, Complete_data_y_train.values)
y_pred = knn.predict(Complete_data_X_test.values)
conf_matrix = ConfusionMatrix(actual_vector=np.array(Complete_data_y_test), predict_vector=np.array(y_pred))
print(conf_matrix)

# GaussianNB
print("GaussianNB Models")
creat_model_with_cross_validation("GaussianNB_CrossModel",GaussianNB(),Complete_data_X,data_targets_y,3)
creat_model_without_cross_validation("GaussianNB_Model", GaussianNB() )

# GaussianNB
print("Decision Tree Classifier Models")
creat_model_with_cross_validation("DecisionTreeClassifier_CrossModel",DecisionTreeClassifier(),Complete_data_X,data_targets_y,3)
creat_model_without_cross_validation("DecisionTreeClassifier_Model", DecisionTreeClassifier() )

# Logistic Regression
print("Logistic Regression Models")
creat_model_with_cross_validation("Lregression_CrossModel",LogisticRegression(),Complete_data_X,data_targets_y)
creat_model_without_cross_validation("Lregression_Model",LogisticRegression())


# Ensemble models bagging
print("Extra Trees Classifier Models")
creat_model_with_cross_validation("ExtraTreesClassifier_CrossModel",ExtraTreesClassifier(),Complete_data_X,data_targets_y)
creat_model_without_cross_validation("ExtraTreesClassifier_Model",ExtraTreesClassifier())
# create_models(ETC, "ExtraTreesClassifier")

print("Bagging Classifier Models")
baggingC = BaggingClassifier()
creat_model_with_cross_validation("BaggingClassifier_CrossModel",BaggingClassifier(),Complete_data_X,data_targets_y)
creat_model_without_cross_validation("BaggingClassifier_Model",BaggingClassifier())

print("Random Forest Classifier Models")
creat_model_with_cross_validation("RandomForestClassifier_CrossModel",RandomForestClassifier(),Complete_data_X,data_targets_y)
creat_model_without_cross_validation("RandomForestClassifier_Model",RandomForestClassifier())


# Ensemble model boosting
print("Gradient Boosting Classifier Models")
creat_model_with_cross_validation("GradientBoostingClassifier_CrossModel",GradientBoostingClassifier(),Complete_data_X,data_targets_y)
creat_model_without_cross_validation("GradientBoostingClassifier_Model",GradientBoostingClassifier())

print("AdaBoost Classifier Models")
creat_model_with_cross_validation("GradientBoostingClassifier_CrossModel",AdaBoostClassifier(),Complete_data_X,data_targets_y)
creat_model_without_cross_validation("GradientBoostingClassifier_Model",AdaBoostClassifier())

print("Hist Gradient Boosting Classifier Models")
creat_model_with_cross_validation("HistGradientBoostingClassifier_CrossModel",HistGradientBoostingClassifier(),Complete_data_X,data_targets_y)
creat_model_without_cross_validation("GradientBoostingClassifier_Model",HistGradientBoostingClassifier())


# Clustering
print("k Nearest neighbors Models")
Complete_data_y_test_m = Complete_data_y_test - 1
Complete_data_y_train_m = Complete_data_y_train - 1
data_targets_y_m = data_targets_y -1
creat_model_with_cross_validation("KMeans3c_CrossModel", KMeans(n_clusters=2, random_state=0, n_init="auto"),Complete_data_X,data_targets_y_m)
creat_model_without_cross_validation("KMeans3c_Model", KMeans(n_clusters=2, random_state=0, n_init="auto"),y_Test=Complete_data_y_test_m,Y_Train=Complete_data_y_train_m)

creat_model_with_cross_validation("KMeans4c_CrossModel", KMeans(n_clusters=4, random_state=0, n_init="auto"),Complete_data_X,data_targets_y_m)
creat_model_without_cross_validation("KMeans4c_Model", KMeans(n_clusters=4, random_state=0, n_init="auto"),y_Test=Complete_data_y_test_m,Y_Train=Complete_data_y_train_m)


print("Gaussian Mixture Models")
creat_model_with_cross_validation("GaussianMixture3c_CrossModel", GaussianMixture(n_components=3),Complete_data_X,data_targets_y_m)
creat_model_without_cross_validation("KMeans3c_Model", GaussianMixture(n_components=3),y_Test=Complete_data_y_test_m,Y_Train=Complete_data_y_train_m)

creat_model_with_cross_validation("GaussianMixture3c_CrossModel", GaussianMixture(n_components=4),Complete_data_X,data_targets_y_m)
creat_model_without_cross_validation("KMeans3c_Model", GaussianMixture(n_components=4),y_Test=Complete_data_y_test_m,Y_Train=Complete_data_y_train_m)


print("DBSCAN Models")
dbscan = DBSCAN(eps=29, min_samples=2)
dbscan.fit(Complete_data_X_train)
y_pred = dbscan.fit_predict(Complete_data_X_test)
actual_t = Complete_data_y_test-1
print( ConfusionMatrix( actual_vector=np.array(actual_t), predict_vector=np.array(y_pred) ) )


print("Print Neural Network Model")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# Load your dataset (assuming it's in CSV format)
complete_data = pd.read_csv("Complete_treated_dataset")
y = data_targets_y - 1

# Split the dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Complete_data, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Flatten(input_shape=Complete_data.shape[1:]),     # Flatten the input shape
    # Dense(128, activation='relu'),
    # Dense(3, activation='softmax')
    Dense(2005, activation="selu"),
    Dense(2005, activation='relu'),
    Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, validation_split=0.2)  # Assuming you want to use 20% of data for validation

y_pred = model.predict(X_test)
classes_previstas = np.argmax(y_pred, axis=1).astype(int)

# evaluate
conf_matrix = ConfusionMatrix(actual_vector=np.array(y_test.astype(int)), predict_vector=np.array(classes_previstas))
print(f'Test accuracy: {conf_matrix}')