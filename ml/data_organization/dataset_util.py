import pandas as pd

from ml.data_organization.dataloader import Dataset_VariablePercentOddsRatio
from torch.utils.data import DataLoader


def create_train_test_datasets(df, split=0.8):
    df = df.sample(frac=1)
    split_idx = int(len(df.index) * split)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def create_kfold_data(df, kfold):
    split_size = int(len(df.index) / len(kfold))
    split_idxs = [i * split_size for i in range(len(kfold))]
    
    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()
    test_df = pd.DataFrame()

    for idx, fold in zip(split_idxs, kfold):
        if fold == 'tr':
            train_df = pd.concat([train_df, df.iloc[idx:idx+split_size]])
        elif fold == 'v':
            valid_df = pd.concat([valid_df, df.iloc[idx:idx+split_size]])
        elif fold == 'te':
            test_df = pd.concat([test_df, df.iloc[idx:idx+split_size]])

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, valid_df, test_df


def create_kfold_datasets(input_cols, nn_config, train_data, valid_data, test_data):
    train_dataset = Dataset_VariablePercentOddsRatio(train_data, 
                                                    input_cols, 
                                                    timeout=nn_config['trade_timeout'], 
                                                    profit_pct=nn_config['profit_pct'], 
                                                    stop_loss_pct=nn_config['stop_loss_pct'], 
                                                    transforms=nn_config['transforms'])
    valid_dataset = Dataset_VariablePercentOddsRatio(valid_data, 
                                                    input_cols, 
                                                    timeout=nn_config['trade_timeout'], 
                                                    profit_pct=nn_config['profit_pct'], 
                                                    stop_loss_pct=nn_config['stop_loss_pct'], 
                                                    transforms=nn_config['transforms'])
    test_dataset =  Dataset_VariablePercentOddsRatio(test_data, 
                                                    input_cols, 
                                                    timeout=nn_config['trade_timeout'], 
                                                    profit_pct=nn_config['profit_pct'], 
                                                    stop_loss_pct=nn_config['stop_loss_pct'], 
                                                    transforms=nn_config['transforms'])
    return train_dataset, valid_dataset, test_dataset


# TODO: maybe still need the normalization transform?
def create_kfold_datasets_forest(input_cols, train_data, valid_data, test_data):
    train_dataset = Dataset_VariablePercentOddsRatio(train_data, 
                                                    input_cols)
    valid_dataset = Dataset_VariablePercentOddsRatio(valid_data, 
                                                    input_cols)
    test_dataset =  Dataset_VariablePercentOddsRatio(test_data, 
                                                    input_cols)
    return train_dataset, valid_dataset, test_dataset


def create_kfold_dataloaders(nn_config, train_dataset, valid_dataset, test_dataset, shuffle=False):
    train_loader = DataLoader(train_dataset, batch_size=nn_config['batch_size'], shuffle=shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=nn_config['batch_size'], shuffle=shuffle)
    test_loader  = DataLoader(test_dataset, batch_size=nn_config['batch_size'], shuffle=shuffle)
    return train_loader, valid_loader, test_loader
