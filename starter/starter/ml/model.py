from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import joblib
import os
import pandas as pd

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, X_test, y_test):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    X_train_df = pd.DataFrame(X_train)
    y_train_df = pd.DataFrame(y_train)
    X_test_df = pd.DataFrame(X_test)
    y_test_df = pd.DataFrame(y_test)

    kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    pred_test_full =0
    cv_score =[]
    i=1

    for train_index,test_index in kf.split(X_train,y_train):
        print('{} of KFold {}'.format(i,kf.n_splits))
        xtr,xvl = X_train_df.loc[train_index],X_train_df.loc[test_index]
        ytr,yvl = y_train_df.loc[train_index],y_train_df.loc[test_index]
        
        #model
        lr = LogisticRegression(max_iter=1000)
        lr.fit(xtr,ytr)
        score = roc_auc_score(yvl,lr.predict(xvl))
        print('ROC AUC score:',score)
        cv_score.append(score)    
        pred_test = lr.predict_proba(X_test)[:,1]
        pred_test_full +=pred_test
        i+=1
    return lr



def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

def save_model(model, path):
    if os.path.exists(path):
        joblib.dump(model, f"{path}/model.joblib")
    else:
        os.makedirs(path)
        joblib.dump(model, f"{path}/model.joblib")