class training_hyperparameters:
    kfold = 5
    undersample_technique = "random"
    oversample_technique = "random"



# Logistic Regression Model
param_grid_logregression = {"penalty" : ['l2'],
              "C" : [1, 10],
             "max_iter" : [1000]}



#Random Forest Model
param_grid_rf = {"n_estimators": [10, 100, 1000],
                 "n_jobs":[-1]}