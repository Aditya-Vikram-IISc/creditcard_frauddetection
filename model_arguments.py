class training_parameters:
    kfold = 5



# Logistic Regression Model
param_grid_logregression = {"penalty" : ['l2'],
              "C" : [1, 10],
             "max_iter" : [1000]}