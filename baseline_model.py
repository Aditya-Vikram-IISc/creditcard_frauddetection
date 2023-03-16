import pandas as pd
from sklearn.linear_model import LogisticRegression
from utilities import splitdf_train_test, stratify_kfold
from sampling import normal_training
from model_arguments import param_grid_logregression

# read the df
df = pd.read_csv("creditcard.csv")

# split the df into train test split. Training will be done via 5-fold CV and finally tested on test dataset
train_df, test_df = splitdf_train_test(df, statify_col="Class", test_size= 0.2, random_state=0)

# genrate a kfold df from train_df
stratified_traindf = stratify_kfold(train_df, statify_col= "Class", k=5)

# train the model on Logistic Regression
log_model = normal_training(stratified_traindf, ml_model = LogisticRegression, kfolds=5, param_grid= param_grid_logregression)
