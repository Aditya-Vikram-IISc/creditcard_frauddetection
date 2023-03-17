class training_hyperparameters:
    kfold = 5
    undersample_technique = "random"
    oversample_technique = "random"



# Logistic Regression Model
param_grid_logregression = {"penalty" : ['l2'],
              "C" : [1, 10],
             "max_iter" : [1000]}