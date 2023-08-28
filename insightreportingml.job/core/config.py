# Application Settings
# From the enviornments vaiables

from pydantic import Field, BaseSettings

class ApplicationSettings(BaseSettings):
    SEQ_API_KEY: str = Field(default='',env='SEQ_API_KEY')
    SEQ_SERVER: str = Field(default='',env='SEQ_SERVER')
    MLMODEL_DIRECTORY: str = Field(default='', env='FileStoreSettings__StorageDirectory')

class Settings(ApplicationSettings):
   
    MLMODEL_PATH: str = '/mlmodels'
    MLMODEL_ALLFIELDS_NAME = 'model_allfields.pkl'
    MLMODEL_TEXTONLY_NAME = 'model_textonly.pkl'

    MLMODEL_ALLFIELDS_VECTOR_NAME = 'model_allfields_vectorizer.joblib'
    MLMODEL_TEXTONLY_VECTOR_NAME = 'model_textonly_vectorizer.joblib'
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True

settings = Settings()