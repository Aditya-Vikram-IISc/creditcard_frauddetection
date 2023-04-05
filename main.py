import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from utilities import splitdf_train_test, random_binary_oversample
from model_arguments import training_hyperparameters
from sklearn.metrics import recall_score, precision_score, average_precision_score


# get the hyperparameters/ arguments
training_hyperparams = training_hyperparameters()
rf_parameters = {"n_estimators" : 100, "n_jobs" : -1}

# read the df
df = pd.read_csv("creditcard.csv")

# split the df into train test split. Training will be done via 5-fold CV and finally tested on test dataset
train_imb_df, test_df = splitdf_train_test(df, statify_col="Class", test_size= 0.15, random_state=0)

# oversample the training dataset
train_df = random_binary_oversample(train_imb_df, category_col="Class", random_state=121)

# get train and test arrays
train_X = train_df.drop(["Class"], axis=1).values
train_y = train_df["Class"].values
test_X = test_df.drop(["Class"], axis=1).values
test_y = test_df["Class"].values

# Train the model on the selected model : RF {'n_estimators': 100} + Oversampling
model = RandomForestClassifier(**rf_parameters)
model.fit(train_X, train_y)

# predict the results
train_y_predicted_proba = model.predict_proba(train_X)[:, 1]
test_y_predicted_proba = model.predict_proba(test_X)[:, 1]

train_y_predicted = np.where(train_y_predicted_proba >= training_hyperparams.threshold, 1, 0)
test_y_predicted = np.where(test_y_predicted_proba >= training_hyperparams.threshold, 1, 0)

# get the results
train_recall = recall_score(train_y, train_y_predicted)
train_precision = precision_score(train_y, train_y_predicted)
train_auprc = average_precision_score(train_y, train_y_predicted_proba)
test_recall = recall_score(test_y, test_y_predicted)
test_precision = precision_score(test_y, test_y_predicted)
test_auprc = average_precision_score(test_y, test_y_predicted_proba)

print(f"Training Recall: {train_recall}")
print(f"Training Precision: {train_precision}")
print(f"Training AUPRC: {train_auprc}")
print(f"Test Recall: {test_recall}")
print(f"Test Precision: {test_precision}")
print(f"Test AUPRC: {test_auprc}")


# save the model weights
model_path = "weights/rf_100_oversample.joblib"
# save model
joblib.dump(model, model_path)
# # load model
# loaded_model = joblib.load(model_path)