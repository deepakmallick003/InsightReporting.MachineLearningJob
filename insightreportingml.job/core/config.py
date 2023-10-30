# Application Settings
# From the enviornments vaiables

from pydantic import Field, BaseSettings

class ApplicationSettings(BaseSettings):
    SEQ_API_KEY: str = Field(default='',env='SEQ_API_KEY')
    SEQ_SERVER: str = Field(default='',env='SEQ_SERVER')
    MLMODEL_DIRECTORY: str = Field(default='', env='FileStoreSettings__StorageDirectory')
    CONN_STRING: str = Field(default='', env='ConnectionStrings__insightreporting')

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
    def parsed_connection_string(self):
        components = dict(item.split('=') for item in self.CONN_STRING.split(';') if item)
        return {
            'database': components.get('database', ''),
            'username': components.get('username', ''),
            'password': components.get('password', ''),
            'server': components.get('server', '')
        }

    @property
    def pyodbc_connection_string(self):
        parsed = self.parsed_connection_string
        return (f"DRIVER={self.DB_DRIVER};"
                f"SERVER={parsed['server']};"
                f"DATABASE={parsed['database']};"
                f"UID={parsed['username']};"
                f"PWD={parsed['password']};")

settings = Settings()