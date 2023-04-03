import numpy as np
from utilities import param_combinations, random_binary_undersample, random_binary_oversample, model_output_plot
from model_arguments import training_hyperparameters
from sklearn.metrics import recall_score, precision_score, average_precision_score, precision_recall_curve

training_hyperparams = training_hyperparameters()


def normal_training(df, ml_model, param_grid, kfolds = training_hyperparams.kfold, threshold =0.5):
    #Get all possible combinations of parameters
    all_combinations_ = param_combinations(param_dict=param_grid)

    #Train the model. For each param, perform k-fold
    for param_ in all_combinations_:
        # Get a list for storing training and validation metrics
        train_recall_list = []
        train_precision_list = []
        train_auprc_list = []
        valid_recall_list = []
        valid_precision_list = []
        valid_auprc_list = []

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
            train_y_predicted_proba = model_.predict_proba(train_X)[:,1]
            valid_y_predicted_proba = model_.predict_proba(valid_X)[:,1]

            train_y_predicted = np.where(train_y_predicted_proba >= threshold, 1, 0)
            valid_y_predicted = np.where(valid_y_predicted_proba >= threshold, 1, 0)

            #append the results
            train_recall_list.append(recall_score(train_y, train_y_predicted))
            train_precision_list.append(precision_score(train_y, train_y_predicted))
            train_auprc_list.append(average_precision_score(train_y, train_y_predicted_proba))
            valid_recall_list.append(recall_score(valid_y, valid_y_predicted))
            valid_precision_list.append(precision_score(valid_y, valid_y_predicted))
            valid_auprc_list.append(average_precision_score(valid_y, valid_y_predicted_proba))

            # plot the training and validation plots for the last iteration of each model+hp combination
            if f ==0:
                precisionT, recallT, thresholdsT = precision_recall_curve(train_y, train_y_predicted_proba)
                precisionV, recallV, thresholdsV = precision_recall_curve(valid_y, valid_y_predicted_proba)
                thresholdsT = np.append(thresholdsT, [1])
                thresholdsV = np.append(thresholdsV, [1])

                fig_title = f"{model_.__class__.__name__}_{param_}"
                path = f"figures_and_charts/{model_.__class__.__name__}_normal_{''.join(str(x) + '_' for x in param_.values())}.png"
                model_output_plot(train_pred_prob = train_y_predicted_proba, precisionT = precisionT, recallT = recallT, thresholdsT = thresholdsT, \
                                  valid_pred_prob = valid_y_predicted_proba, precisionV = precisionV, recallV = recallV, thresholdsV= thresholdsV, \
                                  title = fig_title, path= path, tr_auprc = average_precision_score(train_y, train_y_predicted_proba), \
                                  val_auprc = average_precision_score(valid_y, valid_y_predicted_proba))

        print(f"Model Parameters: {param_}")
        print(f"Training: Mean Recall: {np.mean(train_recall_list)}, Std Recall: {np.std(train_recall_list)}")
        print(f"Training: Mean Precision: {np.mean(train_precision_list)}, Std Precision: {np.std(train_precision_list)}")
        print(f"Training: AUPRC: {np.mean(train_auprc_list)}, Std Precision: {np.std(train_auprc_list)}")
        print(f"Validation: Mean Recall: {np.mean(valid_recall_list)}, Std Precision: {np.std(valid_recall_list)}")
        print(f"Validation: Mean Precision: {np.mean(valid_precision_list)}, Std Precision: {np.std(valid_precision_list)}")
        print(f"Validation: AUPRC: {np.mean(valid_auprc_list)}, Std Precision: {np.std(valid_auprc_list)}")


def undersample_training(df, ml_model, param_grid, kfolds = training_hyperparams.kfold, \
                         undersample_technique = training_hyperparams.undersample_technique, threshold = 0.5):
    #Get all possible combinations of the parameters
    all_combinations_ = param_combinations(param_dict=param_grid)

    #Train the model. For each param, perform k-fold
    for param_ in all_combinations_:
        # Get a list for storing training and validation metrics
        train_recall_list = []
        train_precision_list = []
        train_auprc_list = []
        valid_recall_list = []
        valid_precision_list = []
        valid_auprc_list = []

        for f in range(kfolds):
            #create a model instance
            model_ = ml_model(**param_)

            ## get train and validation dfs
            # replace imbalanced df with balanced df for training
            train_imb_df = df[df["k_fold"] != f].sample(frac=1).reset_index(drop=True)
            if undersample_technique == "random":
                train_df = random_binary_undersample(train_imb_df, category_col="Class", random_state=121)
            else:
                train_df = random_binary_undersample(train_imb_df, category_col="Class", random_state=121)
# todo: Right now else statement is same as if statement. Change it basis need
            valid_df = df[df["k_fold"] == f].sample(frac=1).reset_index(drop=True)

            #get train and validation arrays
            train_X = train_df.drop(["k_fold", "Class"], axis=1).values
            train_y = train_df["Class"].values
            valid_X = valid_df.drop(["k_fold", "Class"], axis=1).values
            valid_y = valid_df["Class"].values

            #train the model
            model_.fit(train_X, train_y)

            #predict the results
            train_y_predicted_proba = model_.predict_proba(train_X)[:,1]
            valid_y_predicted_proba = model_.predict_proba(valid_X)[:,1]

            train_y_predicted = np.where(train_y_predicted_proba >= threshold, 1, 0)
            valid_y_predicted = np.where(valid_y_predicted_proba >= threshold, 1, 0)

            #append the results
            train_recall_list.append(recall_score(train_y, train_y_predicted))
            train_precision_list.append(precision_score(train_y, train_y_predicted))
            train_auprc_list.append(average_precision_score(train_y, train_y_predicted_proba))
            valid_recall_list.append(recall_score(valid_y, valid_y_predicted))
            valid_precision_list.append(precision_score(valid_y, valid_y_predicted))
            valid_auprc_list.append(average_precision_score(valid_y, valid_y_predicted_proba))

            # plot the training and validation plots for the last iteration of each model+hp combination
            if f ==0:
                precisionT, recallT, thresholdsT = precision_recall_curve(train_y, train_y_predicted_proba)
                precisionV, recallV, thresholdsV = precision_recall_curve(valid_y, valid_y_predicted_proba)
                thresholdsT = np.append(thresholdsT, [1])
                thresholdsV = np.append(thresholdsV, [1])

                fig_title = f"{model_.__class__.__name__}_{param_}"
                path = f"figures_and_charts/{model_.__class__.__name__}_undersample_{''.join(str(x) + '_' for x in param_.values())}.png"
                model_output_plot(train_pred_prob = train_y_predicted_proba, precisionT = precisionT, recallT = recallT, thresholdsT = thresholdsT, \
                                  valid_pred_prob = valid_y_predicted_proba, precisionV = precisionV, recallV = recallV, thresholdsV= thresholdsV, \
                                  title = fig_title, path= path, tr_auprc = average_precision_score(train_y, train_y_predicted_proba), \
                                  val_auprc = average_precision_score(valid_y, valid_y_predicted_proba))

        print(f"Model Parameters: {param_}")
        print(f"Training: Mean Recall: {np.mean(train_recall_list)}, Std Recall: {np.std(train_recall_list)}")
        print(f"Training: Mean Precision: {np.mean(train_precision_list)}, Std Precision: {np.std(train_precision_list)}")
        print(f"Training: AUPRC: {np.mean(train_auprc_list)}, Std Precision: {np.std(train_auprc_list)}")
        print(f"Validation: Mean Recall: {np.mean(valid_recall_list)}, Std Precision: {np.std(valid_recall_list)}")
        print(f"Validation: Mean Precision: {np.mean(valid_precision_list)}, Std Precision: {np.std(valid_precision_list)}")
        print(f"Validation: AUPRC: {np.mean(valid_auprc_list)}, Std Precision: {np.std(valid_auprc_list)}")


def oversample_training(df, ml_model, param_grid, kfolds = training_hyperparams.kfold, \
                        oversample_technique = training_hyperparams.oversample_technique, threshold =0.5):
    #Get all possible combinations of the parameters
    all_combinations_ = param_combinations(param_dict=param_grid)

    #Train the model. For each param, perform k-fold
    for param_ in all_combinations_:
        # Get a list for storing training and validation metrics
        train_recall_list = []
        train_precision_list = []
        train_auprc_list = []
        valid_recall_list = []
        valid_precision_list = []
        valid_auprc_list = []

        for f in range(kfolds):
            #create a model instance
            model_ = ml_model(**param_)

            ## get train and validation dfs
            # replace imbalanced df with balanced df for training
            train_imb_df = df[df["k_fold"] != f].sample(frac=1).reset_index(drop=True)
            if oversample_technique == "random":
                train_df = random_binary_oversample(train_imb_df, category_col="Class", random_state=121)
            else:
                train_df = random_binary_oversample(train_imb_df, category_col="Class", random_state=121)
# todo: Righht now else statement is same as if statement. Change it basis need

            valid_df = df[df["k_fold"] == f].sample(frac=1).reset_index(drop=True)

            #get train and validation arrays
            train_X = train_df.drop(["k_fold", "Class"], axis=1).values
            train_y = train_df["Class"].values
            valid_X = valid_df.drop(["k_fold", "Class"], axis=1).values
            valid_y = valid_df["Class"].values

            #train the model
            model_.fit(train_X, train_y)

            #predict the results
            train_y_predicted_proba = model_.predict_proba(train_X)[:,1]
            valid_y_predicted_proba = model_.predict_proba(valid_X)[:,1]

            train_y_predicted = np.where(train_y_predicted_proba >= threshold, 1, 0)
            valid_y_predicted = np.where(valid_y_predicted_proba >= threshold, 1, 0)

            #append the results
            train_recall_list.append(recall_score(train_y, train_y_predicted))
            train_precision_list.append(precision_score(train_y, train_y_predicted))
            train_auprc_list.append(average_precision_score(train_y, train_y_predicted_proba))
            valid_recall_list.append(recall_score(valid_y, valid_y_predicted))
            valid_precision_list.append(precision_score(valid_y, valid_y_predicted))
            valid_auprc_list.append(average_precision_score(valid_y, valid_y_predicted_proba))

            # plot the training and validation plots for the last iteration of each model+hp combination
            if f ==0:
                precisionT, recallT, thresholdsT = precision_recall_curve(train_y, train_y_predicted_proba)
                precisionV, recallV, thresholdsV = precision_recall_curve(valid_y, valid_y_predicted_proba)
                thresholdsT = np.append(thresholdsT, [1])
                thresholdsV = np.append(thresholdsV, [1])

                fig_title = f"{model_.__class__.__name__}_{param_}"
                path = f"figures_and_charts/{model_.__class__.__name__}_oversample_{''.join(str(x) + '_' for x in param_.values())}.png"
                model_output_plot(train_pred_prob = train_y_predicted_proba, precisionT = precisionT, recallT = recallT, thresholdsT = thresholdsT, \
                                  valid_pred_prob = valid_y_predicted_proba, precisionV = precisionV, recallV = recallV, thresholdsV= thresholdsV, \
                                  title = fig_title, path= path, tr_auprc = average_precision_score(train_y, train_y_predicted_proba), \
                                  val_auprc = average_precision_score(valid_y, valid_y_predicted_proba))

        print(f"Model Parameters: {param_}")
        print(f"Training: Mean Recall: {np.mean(train_recall_list)}, Std Recall: {np.std(train_recall_list)}")
        print(f"Training: Mean Precision: {np.mean(train_precision_list)}, Std Precision: {np.std(train_precision_list)}")
        print(f"Training: AUPRC: {np.mean(train_auprc_list)}, Std Precision: {np.std(train_auprc_list)}")
        print(f"Validation: Mean Recall: {np.mean(valid_recall_list)}, Std Precision: {np.std(valid_recall_list)}")
        print(f"Validation: Mean Precision: {np.mean(valid_precision_list)}, Std Precision: {np.std(valid_precision_list)}")
        print(f"Validation: AUPRC: {np.mean(valid_auprc_list)}, Std Precision: {np.std(valid_auprc_list)}")