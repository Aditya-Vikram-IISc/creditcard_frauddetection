# Summary

### Problem Overview:
For a detailed problem statement refer to [kaggle_creditcard_fraud_detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). 
The dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. 
The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.


### Experimentation:
Limited experimentation to Logistic Regression and Random Forest only as significantly good results were achieved
using Random Forest model. 

For each model + hyperparameter combination, we performed a 5-fold stratified cross-validation across three different 
sampling strategies i.e. normal, undersampling and oversampling. **AUPRC for the validation dataset** was chosen as 
the primary criteria. Do not that the sampling strategy (if any) was applied only on the training datasets (i.e 4 out of
5 folds), and the 5-th fold of data was left as is it. Refer to **experimentation.py** and **sampling.py** for more info.

The results of the experiments are summarized below:


| Model Name | Parameters                                   | Sampling                 | Training      |               |               | Validation    |               |             |
|------------|----------------------------------------------|--------------------------|---------------|---------------|---------------|---------------|---------------|-------------|
|            |                                              |                          | Recall        | Precision     | AUPRC         | Recall        | Precision     | AUPRC       |
| LR         | {'penalty': 'l2', 'C': 1, 'max_iter': 1000}  | Same as original dataset | 0.642 ± 0.021 | 0.866 ± 0.017 | 0.699 ± 0.033 | 0.648 ± 0.087 | 0.874 ± 0.037 | 0.686 ± 0.059 |
| LR         | {'penalty': 'l2', 'C': 10, 'max_iter': 1000} | Same as original dataset | 0.647 ± 0.009 | 0.848 ± 0.021 | 0.677 ± 0.023 | 0.636 ± 0.092 | 0.839 ± 0.028 | 0.663 ± 0.081 |
| LR         | {'penalty': 'l2', 'C': 1, 'max_iter': 1000}  | Random Undersampled      |0.913 ± 0.019	|	0.963 ± 0.009	|	0.979 ± 0.007	|	0.902 ± 0.025	|	0.042 ± 0.005	|	0.686 ± 0.044 |
| LR         | {'penalty': 'l2', 'C': 10, 'max_iter': 1000} | Random Undersampled      |0.916 ± 0.011|0.965 ± 0.008|0.983 ± 0.006|0.912 ± 0.023|0.042 ± 0.007|0.673 ± 0.079|
| LR         | {'penalty': 'l2', 'C': 1, 'max_iter': 1000}  | Random Oversampled       |0.918 ± 0.017|0.965 ± 0.005|0.983 ± 0.007|0.914 ± 0.019|0.046 ± 0.005|0.714 ± 0.031|
| LR         | {'penalty': 'l2', 'C': 10, 'max_iter': 1000} | Random Oversampled       |0.91 ± 0.018|0.96 ± 0.005|0.98 ± 0.007|0.914 ± 0.031|0.041 ± 0.005|0.711 ± 0.023|
| RF         | {'n_estimators': 10}                         | Same as original dataset | 0.987 ± 0.005|0.994 ± 0.004|0.999 ± 0|0.799 ± 0.037|0.928 ± 0.032|0.821 ± 0.034|
| RF         | {'n_estimators': 100}                        | Same as original dataset |1 ± 0|1 ± 0|1 ± 0|0.797 ± 0.039|0.947 ± 0.031|0.855 ± 0.032|
| RF         | {'n_estimators': 1000}                       | Same as original dataset |1 ± 0|1 ± 0|1 ± 0|0.792 ± 0.038|0.947 ± 0.031|0.845 ± 0.031|
| RF         | {'n_estimators': 10}                         | Random Undersampling     |0.918 ± 0.017|1 ± 0|1 ± 0|0.792 ± 0.038|0.029 ± 0.006|0.551 ± 0.082|
| RF         | {'n_estimators': 100}                        | Random Undersampling     |0.999 ± 0.001|1 ± 0|1 ± 0|0.904 ± 0.044|0.061 ± 0.012|0.748 ± 0.063|
| RF         | {'n_estimators': 1000}                       | Random Undersampling     |1 ± 0|1 ± 0|1 ± 0|0.916 ± 0.043|0.059 ± 0.021|0.746 ± 0.062|
| RF         | {'n_estimators': 10}                         | Random Oversampling      |1 ± 0|1 ± 0|1 ± 0|0.797 ± 0.047|0.927 ± 0.019|0.831 ± 0.047|
| RF         | {'n_estimators': 100}                        | Random Oversampling      |1 ± 0|1 ± 0|1 ± 0|0.792 ± 0.044|0.948 ± 0.012|0.861 ± 0.048|
| RF         | {'n_estimators': 1000}                       | Random Oversampling      |1 ± 0|1 ± 0|1 ± 0|0.79 ± 0.045|0.943 ± 0.014|0.865 ± 0.047|

The final model selected is RF  {'n_estimators': 1000} .

### Results:



### Things to do:
1. Feature engineering
2. Probabiility calibration
