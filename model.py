from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV


class model:
    def __init__(self):
        # defining the classifier variable
        self.classifier = None

        # defining the textual columns' and numeric columns' headings
        self.textual_columns = ["title", "location", "description"]
        self.numeric_columns = ["telecommuting", "has_company_logo", "has_questions"]

    def train(self, data):
        x = data.drop(['fraudulent'], axis=1)
        y = data["fraudulent"]

        # fitting and transforming the textual column data
        text_cols = []
        for column in self.textual_columns:
            preprocessor = TfidfVectorizer(stop_words="english", norm="l2", use_idf=False, smooth_idf=False)
            text_col = preprocessor.fit_transform(x[column])
            text_col = text_col.toarray()
            text_cols.append(text_col)
            setattr(self, f"preprocessor_{column}", preprocessor)

        # fitting and transforming the numeric column data
        num_cols = x[self.numeric_columns].to_numpy()

        # concatenating textual and numeric columns together
        data_frame = np.hstack(text_cols + [num_cols])
        data_frame = pd.DataFrame(data_frame)

        # using SGDClassifier
        method = SGDClassifier()
        decision_keys = {"loss": ("hinge", "log_loss", "perceptron"), "penalty": ("l2", "l1"), "alpha": [0.0001, 0.01]}
        self.classifier = RandomizedSearchCV(method, decision_keys, cv=5)
        self.classifier.fit(data_frame, y)

    def predict(self, data):
        x = data.drop(['fraudulent'], axis=1)

        # transforming the textual column data based on the stored preprocessor attribute
        text_cols = []
        for column in self.textual_columns:
            preprocessor = getattr(self, f"preprocessor_{column}")
            text_col = preprocessor.transform(x[column])
            text_col = text_col.toarray()
            text_cols.append(text_col)

        # fitting and transforming the numeric column data
        num_cols = x[self.numeric_columns].to_numpy()

        # concatenating textual and numeric columns together
        data_frame = np.hstack(text_cols + [num_cols])
        data_frame = pd.DataFrame(data_frame)

        # predicting the values for fraudulent column
        predictions = self.classifier.predict(data_frame)
        return predictions
