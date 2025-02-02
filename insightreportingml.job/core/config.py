# Application Settings
# From the enviornments vaiables

from pydantic import Field, BaseSettings

class ApplicationSettings(BaseSettings):
    SEQ_API_KEY: str = Field(default='',env='SEQ_API_KEY')
    SEQ_SERVER: str = Field(default='',env='SEQ_SERVER')
    MLMODEL_DIRECTORY: str = Field(default='', env='FileStoreSettings__StorageDirectory')
    DB_SERVER: str = Field(default='', env='InsightReporting_DB_SERVER')
    DB_NAME: str = Field(default='', env='InsightReporting_DB_NAME')
    DB_UID: str = Field(default='', env='InsightReporting_DB_UID')
    DB_PWD: str = Field(default='', env='InsightReporting_DB_PWD')

class Settings(ApplicationSettings):
   
    MLMODEL_PATH: str = '/mlmodels'
    MLMODEL_ALLFIELDS_NAME = 'RFC Model'
    MLMODEL_TEXTONLY_NAME = 'TF-IDF Model'
    MLMODEL_MODEL_SAVED_AS_NAME='model.pkl'
    MLMODEL_VECTORIZER_SAVED_AS_NAME='vectorizer.joblib'
    
    DB_DRIVER='{ODBC Driver 17 for SQL Server}'
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True

    @property
    def pyodbc_connection_string(self):
        return (f"DRIVER={self.DB_DRIVER};"
                f"SERVER={self.DB_SERVER};"
                f"DATABASE={self.DB_NAME};"
                f"UID={self.DB_UID};"
                f"PWD={self.DB_PWD};")

settings = Settings()