import pandas as pd
import os
import pickle
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

from imblearn.over_sampling import RandomOverSampler

from core.config import settings


class ReTraining:

    wd = os.getcwd()

    def read_from_excel(self):
        data_path = self.wd + settings.MLMODEL_PATH + \
            '/' + settings.MLMODEL_DIRECTORY + '/training-data/data.xlsx'

        df_train = pd.read_excel(data_path)
        df_train.head()

        return df_train

    def save_models(self, version, lr_tfidf, tfidf_vectorizer, rfc_model, preprocessing):
        model_path = self.wd + settings.MLMODEL_PATH + \
            '/' + settings.MLMODEL_DIRECTORY + '/' + version + '/'
        
        if(os.path.exists(model_path) == False):
            os.mkdir(model_path)

        with open(model_path + settings.MLMODEL_TEXTONLY_NAME, 'wb') as file:
            pickle.dump(lr_tfidf, file)

        with open(model_path + settings.MLMODEL_TEXTONLY_VECTOR_NAME, 'wb') as file:
            joblib.dump(tfidf_vectorizer, file)

        with open(model_path + settings.MLMODEL_ALLFIELDS_NAME, 'wb') as file:
            pickle.dump(rfc_model, file)

        with open(model_path + settings.MLMODEL_ALLFIELDS_VECTOR_NAME, 'wb') as file:
            joblib.dump(preprocessing, file)

    def generate_models(self, df_train):

        # separate X from y data
        y_column = 'label'
        y = df_train[y_column]
        X = df_train[df_train.columns.drop([y_column])]

        # over sample
        oversample = RandomOverSampler(sampling_strategy=0.6)
        X_over, y_over = oversample.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_over, y_over, test_size=0.33, random_state=42, shuffle=True)

        cat_column_list = ['iso_language', 'species_name', 'has_geo',
                           'source_name', 'source_country', 'source_region', 'source_subject']

        # store one hot encoder in a pipeline
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        categorical_processing = Pipeline(steps=[('ohe', ohe)])

        # create the ColumnTransormer object
        preprocessing = ColumnTransformer(transformers=[(
            'categorical', categorical_processing, cat_column_list)],  remainder='drop')

        X_train_encoded = preprocessing.fit_transform(X_train)
        X_test_encoded = preprocessing.transform(X_test)

        # fit the model
        rfc_model = RandomForestClassifier(random_state=1)
        rfc_model.fit(X_train_encoded, y_train)

        # make predictions
        yhat = rfc_model.predict(X_test_encoded)

        # evaluate predictions
        acc = accuracy_score(y_test, yhat)

        X_train, X_test, y_train, y_test = train_test_split(df_train["clean_text"],
                                                            df_train[y_column],
                                                            test_size=0.2,
                                                            shuffle=True)

        tfidf_vectorizer = TfidfVectorizer(use_idf=True)

        X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test)

        lr_tfidf = LogisticRegression(solver='liblinear', C=100, penalty='l2')
        lr_tfidf.fit(X_train_vectors_tfidf, y_train)  # model

        y_prob = lr_tfidf.predict_proba(X_test_vectors_tfidf)[:, 1]

        lr_tfidf = LogisticRegression(solver='liblinear', C=100, penalty='l2')
        lr_tfidf.fit(X_train_vectors_tfidf, y_train)  # model

        y_prob = lr_tfidf.predict_proba(X_test_vectors_tfidf)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        return acc, roc_auc, lr_tfidf, tfidf_vectorizer, rfc_model, preprocessing


retraining = ReTraining()
