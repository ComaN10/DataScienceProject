from sklearn.model_selection import KFold
from pycm import ConfusionMatrix
from sklearn.metrics import accuracy_score
class KFoldCrossValidator:
    """
    Performs k-fold cross-validation for model evaluation.
    """
    def __init__(self, k=5):
        """
        Initializes the KFoldCrossValidator object.

        Parameters:
            k (int): Number of folds for cross-validation. Default is 5.
        """
        self.k = k
        self.kf = KFold(n_splits=k, shuffle=True)
        self.cm = None
        self.accuracy_scores = []
        self.sensitivity_scores = []
        self.specificity_scores = []

    def cross_validate(self, model, X, y):
        """
        Performs k-fold cross-validation on the given model using the provided features and labels.

        Parameters:
            model: Machine learning model to be evaluated.
            X (array-like): Features.
            y (array-like): Labels.

        Returns:
            tuple: Average accuracy, sensitivity, and specificity scores.
        """

        avg_sensitivity = 0
        avg_specificity = 0
        for train_index, val_index in self.kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            self.cm = ConfusionMatrix(actual_vector=list(y_val), predict_vector=list(y_pred))

            self.accuracy_scores.append(accuracy_score(y_val, y_pred))
            if self.cm.TPR_Macro != 'None':
                self.sensitivity_scores.append(float(self.cm.TPR_Macro))
            if self.cm.TNR_Macro != 'None':
                self.specificity_scores.append(float(self.cm.TNR_Macro))

        avg_accuracy = sum(self.accuracy_scores) / len(self.accuracy_scores)
        if len(self.sensitivity_scores) != 0:
            avg_sensitivity = sum(self.sensitivity_scores) / len(self.sensitivity_scores)
        if len(self.specificity_scores) != 0:
            avg_specificity = sum(self.specificity_scores) / len(self.specificity_scores)

        print(self.cm)

        return (( "avg_accuracy", avg_accuracy, self.k ) ,
                ( "avg_sensitivity" , avg_sensitivity , len(self.sensitivity_scores) ),
                ( "avg_specificity" ,avg_specificity ,len(self.specificity_scores) ))

    def evaluate_on_test_set(self, model, X_test, y_test):
        """
        Evaluates the trained model on the test set.

        Parameters:
            model: Trained machine learning model.
            X_test (array-like): Test features.
            y_test (array-like): Test labels.

        Returns:
            tuple: Accuracy, sensitivity, and specificity scores on the test set.
        """
        y_pred = model.predict(X_test)
        cm = ConfusionMatrix(actual_vector=list(y_test), predict_vector=list(y_pred))
        accuracy = cm.Overall_ACC
        sensitivity = cm.TPR_Macro
        specificity = cm.TNR_Macro
        return accuracy, sensitivity, specificity