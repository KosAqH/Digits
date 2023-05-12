import torch
import pandas as pd
from sklearn import model_selection
import matplotlib.pyplot as plt

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

def load_data(filename, h, w):
    df = pd.read_csv(filename)
    X, y = reshape_data_for_cnn(df, h, w)

    seed = 0
    X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y, test_size=0.2, random_state=seed)

    train_loader = create_dataloader(X_train, y_train)
    val_loader = create_dataloader(X_val, y_val)
    return train_loader, val_loader

def plot_acc(accs: list) -> None:
    x = list(range(1, len(accs)+1))
    plt.plot(x, accs)