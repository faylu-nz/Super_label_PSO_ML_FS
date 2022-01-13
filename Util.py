import pandas as pd


def read_arff(file):
    with open(file, encoding="utf-8") as f:
        header = []
        for line in f:
            if line.startswith("@attribute"):
                header.append(line.split()[1])
            elif line.startswith("@data"):
                break
        df = pd.read_csv(f, header=None)
        df.columns = header
    return df

# K-fold only accept X, y in numpy form, and return X_train, X_test, y_train, y_test in each loop(split).
# Convert X_train, X_test, y_train, y_test from numpy to dataframe, for super_label calculation call.


def Convert_to_df(X_train, X_test, y_train, y_test):

    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    y_train_df = pd.DataFrame(y_train)
    y_test_df = pd.DataFrame(y_test)

    return X_train_df, X_test_df, y_train_df, y_test_df
