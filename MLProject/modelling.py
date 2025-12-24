import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# TETAP LOCALHOST (default)
dagshub.init(
    repo_owner="BlueNebulaa",
    repo_name="SML_Muhammad-Nabil-Ibrahim",
    mlflow=True
)

df = pd.read_csv("/home/pai/Downloads/msml/WorkflowCI/MLProject/data_clean/data_clean.csv")

y=df['Heart Disease']
    
X_train,X_test=train_test_split(df,test_size=0.25,random_state=42,stratify=y)

Y_train=X_train['Heart Disease']
X_train=X_train.drop(columns=['Heart Disease'])

Y_test=X_test['Heart Disease']
X_test=X_test.drop(columns=['Heart Disease'])

param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear"]
}

grid = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid,
    cv=3
)

with mlflow.start_run(run_name="logreg_manual_tuning"):
    grid.fit(X_train, Y_train)
    model = grid.best_estimator_

    y_pred = model.predict(X_test)

    acc = accuracy_score(Y_test, y_pred)
    prec = precision_score(Y_test, y_pred, pos_label="Presence")
    rec = recall_score(Y_test, y_pred, pos_label="Presence")

    # MANUAL LOGGING
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)

    # ARTEFAK MODEL
    mlflow.sklearn.log_model(model, "model")

    print("Best params:", grid.best_params_)
    
