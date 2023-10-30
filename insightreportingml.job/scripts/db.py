import numpy as np
import pandas as pd
import pyodbc
import json
from core.config import settings

class DatabaseManager:
    def __init__(self, logging):
        self.connection_string = f'DRIVER={settings.DB_DRIVER};SERVER={settings.DB_SQLSERVER};DATABASE={settings.DB_NAME};UID={settings.DB_UID};PWD={settings.DB_PWD};'
        self.logging = logging

    def __enter__(self):
        self.connection = self._connect()
        self.cursor = self.connection.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _connect(self):
        try:
            connection = pyodbc.connect(self.connection_string)
            self.logging.info("Connected to SQL Server")
            return connection
        except Exception as e:
            self.logging.error("Error connecting to SQL Server: "  + str(e))
            return None
        
    def close(self):
            if self.connection:
                self.connection.close()
                self.logging.info("SQL Server connection closed")
        
    def get_unprocessed_model_versions(self):
        try:
            self.cursor.execute('EXEC dbo.GetUnprocessedModelVersions')
            rows = self.cursor.fetchall()
            return rows 
        except pyodbc.Error as e:
            self.logging.error("Error fetching unprocessed model versions from db: " + str(e))
            return None 
        
    def get_training_data(self):
        try:
            result_data = {                
                'promoted_demoted_records': [],  # All promoted/demoted records since last retraining
                'existing_training_records': []  # All existing training data
            }

            self.cursor.execute('EXEC dbo.GetTrainingData')  # Executing the stored procedure

            # First, we will fetch the 'promoted_demoted_records' result set.
            rows = self.cursor.fetchall()
            result_data['promoted_demoted_records'].extend(rows)

            # Now, we check if another result set is present and move to it.
            has_next_set = self.cursor.nextset()

            # If we successfully moved to the next result set, we fetch the 'training_data'.
            if has_next_set:
                rows = self.cursor.fetchall()
                result_data['existing_training_records'].extend(rows)

            return result_data
        except pyodbc.Error as e:
            self.logging.error("Error fetching training data from db: " + str(e))
            return None

    def save_models_to_db(self, best_models_and_data, processing_status=51):
        rfc_model_version_id = best_models_and_data['rfc_model_version_id']
        lr_model_version_id = best_models_and_data['lr_model_version_id']
        
        if processing_status == 1:      
            model_metrics = best_models_and_data['rfc_metrics']
            df_train = best_models_and_data['rfc_training_data']
            self.save_model(processing_status, rfc_model_version_id, model_metrics, df_train)

            model_metrics = best_models_and_data['lr_metrics']
            df_train = best_models_and_data['lr_training_data']
            self.save_model(processing_status, lr_model_version_id, model_metrics, df_train)
        else:
            self.save_model(processing_status, rfc_model_version_id, None, None)
            self.save_model(processing_status, lr_model_version_id, None, None)

    def save_model(self, processing_status, model_version_id, model_metrics=None, df_train=None):
        try:
            cursor = self.connection.cursor()
            training_data_list  = []
            model_metrics_to_save = json.dumps(model_metrics) if model_metrics else ''
        
            if df_train is not None:
                df_train = df_train.rename(columns={
                    'training_data_id': 'TrainingDataID',
                    'item_id': 'ItemID',
                    'clean_text': 'CleanText',
                    'iso_language': 'ISOLanguageCode',
                    'species_name': 'ScientificName',
                    'has_geo': 'GeoRss_Point',
                    'source_name': 'SourceName',
                    'source_country': 'SourceCountry',
                    'source_region': 'SourceRegion',
                    'source_subject': 'SourceSubject',
                    'label': 'Label'
                })     
                
                df_train = df_train.replace({np.nan: None, pd.NA: None})
                df_train['TrainingDataID'] = df_train['TrainingDataID'].apply(self.convert_to_int)
                df_train['ItemID'] = df_train['ItemID'].apply(self.convert_to_int)
                df_train['TrainingDataID'] = df_train['TrainingDataID'].astype('Int64')
                df_train['ItemID'] = df_train['ItemID'].astype('Int64')
                df_train = df_train.replace({np.nan: None, pd.NA: None})

                training_data_list = [tuple(row) for row in df_train.itertuples(index=False)]
                
                tvp_type = [
                    pyodbc.SQL_BIGINT,  # For TrainingDataID
                    pyodbc.SQL_BIGINT,  # For ItemID
                    pyodbc.SQL_WVARCHAR,  # For CleanText
                    pyodbc.SQL_VARCHAR,  # For ISOLanguageCode
                    pyodbc.SQL_VARCHAR,  # For ScientificName
                    pyodbc.SQL_BIT,  # For GeoRss_Point
                    pyodbc.SQL_VARCHAR,  # For SourceName
                    pyodbc.SQL_VARCHAR,  # For SourceCountry
                    pyodbc.SQL_VARCHAR,  # For SourceRegion
                    pyodbc.SQL_VARCHAR,  # For SourceSubject
                    pyodbc.SQL_BIT   # For Label
                ]
                cursor.setinputsizes([tvp_type])

            stored_proc_call = "EXEC dbo.SaveTrainingData ?, ?, ?, ?"
            params = (model_version_id, model_metrics_to_save, processing_status, training_data_list)
            cursor.execute(stored_proc_call,params)
            self.connection.commit()

        except pyodbc.Error as e:
            self.logging.error("Error saving models to db: " + str(e))
            return None

    def convert_to_int(self, value):
        if pd.isna(value): 
            return value  
        else:
            return int(value)