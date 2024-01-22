import pandas as pd


def train(data):
    pass


if __name__ == '__main__':
    # load data
    data = pd.read_csv("data/jobs.csv")

    # replacing missing values with an empty string
    data = data.fillna("")

    # splitting the data into training set and testing set
    # It is divided as 80% training and 20% testing here
    train_data = data.sample(frac=0.8)
    test_data = data.drop(train_data.index)

    train(data)
