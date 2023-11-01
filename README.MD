# Insight Reporting Machine learing JOB
 
## Project Overview
The purpose of the PlantwisePlus.InsightReporting.MicroService.MachineLearningJOB project is to Retrain Machine Learning Model

## Contact Details
* **Technical Contact:** Ravi Bhushan Sinha, Deepak Mallick
* **Business Owner:** Hannah Fiedler
* **Project Manager:** Michelle Jones

## Key Details

### Production URLs
* **Base URL:** N/A

### Development Team(s) or Suppliers
* CABI

### Technology
* **Version:** Python 3.9
* **Logging:** Seq

## Dependencies
* pydantic>=1.8.0,<2.0.0
* seqlog>=0.3.27
* python-dotenv>=1.0.0
* pandas>=2.0.3
* openpyxl>=3.1.2
* imbalanced-learn>=0.11.0
* scikit-learn<= 1.3.2
* spacy>=3.7.2
* nltk>=3.8.1

### Notes
* Create virtual enviornment 
**python -m venv .venv**
* Activate Virtual Environment
**.venv\Scripts\activate**
* Change Project Directory to insightreportingml.job
**cd insightreportingml.job***
* Install dependencies
**pip install -r requirements.txt**
* Run application
**python main.py**
* Run test
**pytest**
* Install other libraries
  * python -m nltk.downloader wordnet
  * python -m nltk.downloader stopwords
  * python -m nltk.downloader punkt
  * python -m nltk.downloader averaged_perceptron_tagger

