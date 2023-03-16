from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import itertools

def splitdf_train_test(df, statify_col = "Class", test_size = 0.2, random_state = 0):
    """
    Parameters:
    df: input df
    statify_col: Column basis which stratify split is to be done
    test_size: size assigned to test dataset while splitting
    random_state: seed to reproduce the results
    Return: train and test dataset
    """
    df_train, df_test = train_test_split(df, test_size = test_size, \
                                         stratify= df[statify_col].values, random_state= random_state)

    # shuffle the datasets
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_test = df_test.sample(frac=1).reset_index(drop=True)
    return df_train, df_test


def stratify_kfold(df, statify_col= "Class", k=5):
    """
    Note: Performs stratified kfold on a given df
    Parameters:
    df - input df
    statify_col: Column basis which stratify split is to be done
    k - number of folds to be done
    Return: stratified kfold df. k indicated in column "k_fold"
    """

    # shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)

    # create a new column and assign it value -1
    df["k_fold"] = -1

    # create an instance of stratified kfold
    stratified_kf = StratifiedKFold(n_splits=k)

    # stratify split the dataframe
    for i, (train_ind, val_ind) in enumerate(stratified_kf.split(df, df[statify_col].values)):
        df.loc[val_ind, "k_fold"] = i

    return df


def param_combinations(param_dict):
    """
    Arguments:
    param_dict - input dict containing various model parameters
    Return: Returns a list of different param combination for the input model
    """
    keys, values = zip(*param_dict.items())
    all_combinations = [dict(zip(keys,v))  for v in itertools.product(*values)]
    return all_combinations
