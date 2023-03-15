from utilities import splitdf_train_test, stratify_kfold
import pandas as pd

# read the df
df = pd.read_csv("creditcard.csv")
# split the df into train test split. Training will be done via 5-fold CV and finally tested on test dataset
train_df, test_df = splitdf_train_test(df, statify_col="Class", test_size= 0.2, random_state=0)

train_df_stra = stratify_kfold(train_df)
print(train_df_stra["k_fold"].value_counts())