import numpy as np
from utilities import param_combinations
from model_arguments import training_parameters
from sklearn.metrics import recall_score, precision_score, average_precision_score

training_params = training_parameters()


def normal_training(df, ml_model, kfolds, param_grid):
    #Get all possible combinations of parameters
    all_combinations_ = param_combinations(param_dict=param_grid)

    #Train the model. For each param, perform k-fold
    for param_ in all_combinations_:
        # Get a list for storing training and validation metrics
        train_recall_list = []
        train_precision_list = []
        valid_recall_list = []
        valid_precision_list = []

        for f in range(kfolds):
            #create a model instance
            model_ = ml_model(**param_)

            #get train and validation dfs
            train_df = df[df["k_fold"] != f].sample(frac=1).reset_index(drop=True)
            valid_df = df[df["k_fold"] == f].sample(frac=1).reset_index(drop=True)

            #get train and validation arrays
            train_X = train_df.drop(["k_fold", "Class"], axis=1).values
            train_y = train_df["Class"].values
            valid_X = valid_df.drop(["k_fold", "Class"], axis=1).values
            valid_y = valid_df["Class"].values

            #train the model
            model_.fit(train_X, train_y)

            #predict the results
            train_y_predicted = model_.predict(train_X)
            valid_y_predicted = model_.predict(valid_X)

            #append the results
            train_recall_list.append(recall_score(train_y, train_y_predicted))
            train_precision_list.append(precision_score(train_y, train_y_predicted))
            valid_recall_list.append(recall_score(valid_y, valid_y_predicted))
            valid_precision_list.append(precision_score(valid_y, valid_y_predicted))

        print(f"Model Parameters: {param_}")
        print(f"Training: Mean Recall: {np.mean(train_recall_list)}, Std Recall: {np.std(train_recall_list)}")
        print(f"Training: Mean Precision: {np.mean(train_precision_list)}, Std Precision: {np.std(train_precision_list)}")
        print(f"Validation: Mean Recall: {np.mean(valid_recall_list)}, Std Precision: {np.mean(valid_recall_list)}")
        print(f"Validation: Mean Precision: {np.mean(valid_precision_list)}, Std Precision: {np.mean(valid_precision_list)}")
