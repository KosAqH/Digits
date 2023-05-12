import torch
import pandas as pd

def reshape_data_for_cnn(df, h, w):
    nrows = df.shape[0]
    X = df.iloc[:, 1:].values.reshape(nrows, 1, h, w)
    y = df.iloc[:, 0].values
    return X, y

def create_dataloader(X, y, shuffle = False, batch_size = 100) -> torch.utils.data.DataLoader:
    X_tensor = torch.Tensor(X)
    y_tensor = torch.Tensor(y).long()
    dataset_train = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=shuffle, batch_size=batch_size)
    return train_loader
