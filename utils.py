def split_dataset(X, Y, share_last):
    """
    Splits data set into two parts.

    Args:
        X (np.array): X data
        Y (np.array): Y data
        share_last (float): how large share of the dataset that should be in the last part of the data set
    """
    split_point = int(len(Y) * share_last)
    X_last = X[:split_point, :]
    X_first = X[split_point:, :]
    Y_last = Y[:split_point]
    Y_first = Y[split_point:]
    return X_first, Y_first, X_last, Y_last
