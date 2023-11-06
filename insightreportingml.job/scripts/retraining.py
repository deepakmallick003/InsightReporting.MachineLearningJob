import re
import string
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
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, auc, roc_curve

from imblearn.over_sampling import RandomOverSampler

from core.config import settings

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class ReTraining:

    def __init__(self):
        self.wd = os.getcwd()
        self.wl = WordNetLemmatizer()

    def read_from_excel(self):
        data_path = self.wd + settings.MLMODEL_PATH + \
            '/' + settings.MLMODEL_DIRECTORY + '/training-data/data.xlsx'

        df_train = pd.read_excel(data_path)
        df_train.head()

        return df_train

    def save_model_to_directory(self, model_id, version, model, vectorizer):
        
        model_name = settings.MLMODEL_ALLFIELDS_NAME if model_id==1 else settings.MLMODEL_TEXTONLY_NAME
        
        model_path = self.wd + settings.MLMODEL_PATH + \
            '/' + settings.MLMODEL_DIRECTORY+ '/' + model_name + '/' + str(version) + '/'

        if(os.path.exists(model_path) == False):
            os.makedirs(model_path)

        with open(model_path + settings.MLMODEL_MODEL_SAVED_AS_NAME, 'wb') as file:
            pickle.dump(model, file)

        with open(model_path + settings.MLMODEL_VECTORIZER_SAVED_AS_NAME, 'wb') as file:
            joblib.dump(vectorizer, file)

    def save_models(self, best_models_meta):
        model_id=best_models_meta['rfc_model_id']
        version=best_models_meta['rfc_model_version_id']
        model=best_models_meta['rfc_model']
        vectorizer=best_models_meta['rfc_vectorizer']
        self.save_model_to_directory(model_id,version, model, vectorizer)
        
        model_id=best_models_meta['lr_model_id']
        version=best_models_meta['lr_model_version_id']
        model=best_models_meta['lr_model']
        vectorizer=best_models_meta['lr_vectorizer']
        self.save_model_to_directory(model_id,version, model, vectorizer)

        print('Best Models Saved to File Storage')

    def generate_rfc_model(self, df_train):

        # separate X from y data
        y_column = 'label'
        y = df_train[y_column]
        X = df_train[df_train.columns.drop([y_column])]

        # over sample
        oversample = RandomOverSampler(sampling_strategy=0.6)
        X_over, y_over = oversample.fit_resample(X, y)

        # split train and test data
        X_train, X_test, y_train, y_test = train_test_split(
            X_over, y_over, test_size=0.33, random_state=42, shuffle=True)

        category_column_list = ['iso_language', 'species_name', 'has_geo',
                           'source_name', 'source_country', 'source_region', 'source_subject']

        # store one hot encoder in a pipeline
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        categorical_processing = Pipeline(steps=[('ohe', ohe)])

        # create the columntransfromer object/vectorizer
        rfc_vectorizer = ColumnTransformer(transformers=[(
            'categorical', categorical_processing, category_column_list)],  remainder='drop')

        #vectorize/encode the train and test data 
        rfc_X_train_encoded = rfc_vectorizer.fit_transform(X_train)
        rfc_X_test_encoded = rfc_vectorizer.transform(X_test)

        #fitting the classification model using random forest classifier
        rfc_model = RandomForestClassifier(random_state=1)
        rfc_model.fit(rfc_X_train_encoded, y_train)

        #predict y value for test dataset
        rfc_y_predict = rfc_model.predict(rfc_X_test_encoded)

        #get model metrics
        model_metrics = self.get_model_metrics(y_test, rfc_y_predict)

        model_details = {
            'metrics': model_metrics,
            'model': rfc_model,
            'vectorizer': rfc_vectorizer,
            'X_train_encoded': rfc_X_train_encoded,
            'X_test_encoded': rfc_X_test_encoded,
            'y_train': y_train,
            'y_test': y_test
        }

        return model_details

    def generate_lr_Model(self, df_train):
        # separate X from y data and split train and test data
        y_column = 'label'
        X_train, X_test, y_train, y_test = train_test_split(df_train["clean_text"],
                                                            df_train[y_column],
                                                            test_size=0.2,
                                                            shuffle=True)

        #vectorize/encode the train and test data with tf-idf
        tfidf_vectorizer = TfidfVectorizer(use_idf=True)
        lr_X_train_encoded = tfidf_vectorizer.fit_transform(X_train)
        lr_X_test_encoded = tfidf_vectorizer.transform(X_test)

        #fitting the classification model using logistic regression
        lr_model = LogisticRegression(solver='liblinear', C=100, penalty='l2')
        lr_model.fit(lr_X_train_encoded, y_train)  # model

        #predict y value for test dataset
        lr_y_predict = lr_model.predict(lr_X_test_encoded)

        #get model metrics
        model_metrics = self.get_model_metrics(y_test, lr_y_predict)

        model_details = {
            'metrics': model_metrics,
            'model': lr_model,
            'vectorizer': tfidf_vectorizer,
            'X_train_encoded': lr_X_train_encoded,
            'X_test_encoded': lr_X_test_encoded,
            'y_train': y_train,
            'y_test': y_test
        }

        return model_details
    
    def get_model_metrics(self, y_test, y_predict):
        accuracy = accuracy_score(y_test, y_predict)
        precision = precision_score(y_test, y_predict) 
        recall = recall_score(y_test, y_predict)
        f1 = f1_score(y_test, y_predict) 
        fpr, tpr, _ = roc_curve(y_test, y_predict)
        area_under_curve = auc(fpr, tpr)
        conf_matrix = confusion_matrix(y_test, y_predict)
        class_report = classification_report(y_test, y_predict, output_dict=True)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': area_under_curve,  
            'classification_report': class_report,  # includes detailed precision, recall, f1-score for each class
            'confusion_matrix': conf_matrix.tolist()  # Convert numpy array to list for JSON serialization  
        }

        return metrics_dict

    def process_training_records(self, records, records_type=1):
        processed_records = []
        for record in records:
            if records_type == 1:    
                cleanText = self.cleantext(record.Title, record.KeywordList)
                label = 1 if record.SelectedByUser == 1 else 0
            else:
                cleanText= record.CleanText
                label =  record.Label

            new_record = {
                'training_data_id': getattr(record, 'TrainingDataID', None),
                'item_id': getattr(record, 'ItemID', None),
                'clean_text': cleanText,
                'iso_language': record.ISOLanguageCode,
                'species_name': record.ScientificName,
                'has_geo': 1 if record.GeoRss_Point == True else 0,
                'source_name': record.SourceName,
                'source_country': record.SourceCountry,
                'source_region': record.SourceRegion,
                'source_subject': record.SourceSubject,
                'label': label
            }
            processed_records.append(new_record)
        return processed_records

    def process_training_data(self, unprocessed_training_data):
        processed_training_data = []
        promoted_demoted_records = unprocessed_training_data['promoted_demoted_records']
        existing_training_records = unprocessed_training_data['existing_training_records']
        processed_training_data.extend(self.process_training_records(promoted_demoted_records, records_type=1))
        processed_training_data.extend(self.process_training_records(existing_training_records, records_type=0))

        # Convert to a DataFrame
        df_train = pd.DataFrame(processed_training_data)
        df_train = df_train[['training_data_id','item_id', 'clean_text', 'iso_language', 'species_name', 'has_geo', 'source_name', 
                            'source_country', 'source_region', 'source_subject', 'label']]
        
        df_train['label'] = df_train['label'].astype('int64')
        
        return df_train

    def get_training_subsets(self, df_train):

        df_new = df_train[df_train['training_data_id'].isna()]  # new records
        df_old = df_train.dropna(subset=['training_data_id'])   # old records

        num_subsets = 3

        # Store the subsets
        training_subsets = []

        # Count of new and old records
        count_new = len(df_new)
        count_old = len(df_old)

        fraction = 0.90  # This would at max trim 10% of old training data
        proposed_total = 0
        while fraction < 1:
            proposed_total = (fraction * count_old) + count_new
            if proposed_total >= count_old:
                break
            else:
                fraction += 0.01

        total_old_records_to_take = int(fraction * count_old)

        for _ in range(num_subsets):
            # Randomly sample old records. Since we're removing some records, the sample size is less than the count_old.
            df_old_sampled = df_old.sample(n=total_old_records_to_take).reset_index(drop=True)
            # Combine new data with the sampled old data
            df_train_subset = pd.concat([df_new, df_old_sampled], ignore_index=True)
            training_subsets.append(df_train_subset)

        training_subsets.append(pd.concat([df_new, df_old], ignore_index=True))

        return training_subsets
       
    def train_models(self, unprocessed_training_data, best_models_meta):

        print('Retraining Started')
        print('Processing Training Data')
        df_train = self.process_training_data(unprocessed_training_data)

        print('Splitting Training Data Into Subsets')
        training_subsets = self.get_training_subsets(df_train)

        # Initialize variables to store the best model and its metadata
        best_models_meta['rfc_model'] = None
        best_models_meta['rfc_vectorizer'] = None
        best_models_meta['rfc_metrics'] = {'accuracy': 0, 'precision': 0, 'auc': 0}
        best_models_meta['rfc_training_data'] = None
        best_models_meta['lr_model'] = None
        best_models_meta['lr_vectorizer'] = None
        best_models_meta['lr_metrics'] = {'accuracy': 0, 'precision': 0, 'auc': 0}
        best_models_meta['lr_training_data'] = None

        print('Starting Retraining with Each Training Subset')
        
        for index, df_train in enumerate(training_subsets, start=1):

            columns_to_drop = ['training_data_id', 'item_id']
            new_df_train = df_train.drop(columns=columns_to_drop)

            print(f'Retraining RFC Model with subset {index}') 
            # Random Forest Classifier
            rfc_model_details = self.generate_rfc_model(new_df_train)

            # Check if the current RFC model is better than the best one so far
            if rfc_model_details['metrics']['accuracy'] > best_models_meta['rfc_metrics']['accuracy']:
                best_models_meta['rfc_metrics'] = rfc_model_details['metrics']
                best_models_meta['rfc_model'] = rfc_model_details['model']
                best_models_meta['rfc_vectorizer'] = rfc_model_details['vectorizer']
                best_models_meta['rfc_training_data'] = df_train

            print(f'Retraining RFC Model completed with subset {index}') 

            print(f'Retraining LR Model completed with subset {index}') 
            # Logistic Regression
            lr_model_details = self.generate_lr_Model(new_df_train)

            # Check if the current RFC model is better than the best one so far
            if lr_model_details['metrics']['accuracy'] > best_models_meta['lr_metrics']['accuracy']:
                best_models_meta['lr_metrics'] = lr_model_details['metrics']
                best_models_meta['lr_model'] = lr_model_details['model']
                best_models_meta['lr_vectorizer'] = lr_model_details['vectorizer']
                best_models_meta['lr_training_data'] = df_train

            print(f'Retraining LR Model completed with subset {index}') 
        
        print('Completed Retraining with Each Training Subset')
        print('Saving Best Models to File Storage')

        self.save_models(best_models_meta) # saves model pickle and joblib to file storage

        print('Returning Model Details to Main Thread')

        return best_models_meta


    #Text-Processing Logics

    def preprocess(self, text):
        text = text.lower()
        text = text.strip()
        text = re.compile('<.*?>').sub('', text)
        text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
        text = re.sub('\s+', ' ', text)
        text = re.sub(r'\[[0-9]*\]', ' ', text)
        text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
        text = re.sub(r'\d', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def stopword(self, string):
        a = [i for i in string.split() if i not in stopwords.words('english')]
        return ' '.join(a)

    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def lemmatizer(self, string):
        word_pos_tags = nltk.pos_tag(word_tokenize(string))

        a = [self.wl.lemmatize(tag[0], self.get_wordnet_pos(tag[1])) for idx, tag in enumerate(
            word_pos_tags)]
        
        return " ".join(a)

    def finalpreprocess(self, string):
        return self.lemmatizer(self.stopword(self.preprocess(string)))

    def cleantext(self, title, keywords):
        clean_title = self.finalpreprocess(title)
        clean_keywords = self.finalpreprocess(keywords)
    
        list_merged = clean_title.split(' ') + clean_keywords.split(' ')
        clean_text = ' '.join(list(dict.fromkeys(list_merged)))

        clean_text = re.sub(r'[^\x00-\x7F]+', '', clean_text)

        return clean_text
    
