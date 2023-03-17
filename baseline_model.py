import pandas as pd
from sklearn.linear_model import LogisticRegression
from utilities import splitdf_train_test, stratify_kfold
from sampling import normal_training, undersample_training, oversample_training
from model_arguments import training_hyperparameters, param_grid_logregression


training_hyperparams = training_hyperparameters()


# read the df
df = pd.read_csv("creditcard.csv")

# split the df into train test split. Training will be done via 5-fold CV and finally tested on test dataset
train_df, test_df = splitdf_train_test(df, statify_col="Class", test_size= 0.2, random_state=0)

# generate a kfold df from train_df
stratified_traindf = stratify_kfold(train_df, statify_col= "Class", k=5)

#### write functions for model training via normal training, undersampling and oversampling

# train the model on Logistic Regression
# normal_training(stratified_traindf, ml_model = LogisticRegression, kfolds=5, param_grid= param_grid_logregression)
# undersample_training(df= stratified_traindf, ml_model = LogisticRegression,\
#                     param_grid = param_grid_logregression, kfolds = training_hyperparams.kfold, \
#                     undersample_technique = training_hyperparams.undersample_technique)

oversample_training(df= stratified_traindf, ml_model = LogisticRegression, \
                    param_grid = param_grid_logregression, kfolds = training_hyperparams.kfold, \
                    oversample_technique = training_hyperparams.oversample_technique)

