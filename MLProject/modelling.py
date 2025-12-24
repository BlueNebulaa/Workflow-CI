import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import accuracy_score
import numpy as np

df = pd.read_csv("/home/pai/Downloads/msml/Eksperimen_SML_Muhammad-nabil-ibrahim/preprocessing/data_clean.csv")

y=df['Heart Disease']
    
X_train,X_test=train_test_split(df,test_size=0.25,random_state=42,stratify=y)

Y_train=X_train['Heart Disease']
X_train=X_train.drop(columns=['Heart Disease'])

Y_test=X_test['Heart Disease']
X_test=X_test.drop(columns=['Heart Disease'])

mlflow.autolog()

with mlflow.start_run(run_name="logistic_regression_basic"):
    model = LogisticRegression(C=0.85,max_iter=1500)
    model.fit(X_train, Y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, y_pred)

    print("Accuracy:", acc)