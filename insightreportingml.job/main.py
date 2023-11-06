import logging
import time

import core.logging
from scripts.retraining import ReTraining
from scripts.db import DatabaseManager

def retraining_job():   
    try:
        logging.info('----------------------')
        print('----------------------')

        logging.info("Retraining Job Iteration Started")
        print("Retraining Job Iteration Started")

        with DatabaseManager() as db_instance:
            logging.info('Fetching Unprocessed Model Versions')
            print('Fetching Unprocessed Model Versions')
            unprocessed_model_versions = db_instance.get_unprocessed_model_versions()
            if unprocessed_model_versions:
                logging.info('Unprocessed Models Versions found, Starting Retraining')
                print('Unprocessed Models Versions found, Starting Retraining')

                best_models_meta= {
                    'rfc_model_version_id': None,
                    'rfc_model_id': None,
                    'lr_model_version_id': None,
                    'lr_model_id': None                    
                }

                for row in unprocessed_model_versions:
                    if row.ModelID == 1:  # RFC model
                        best_models_meta['rfc_model_version_id'] = row.ModelVersionID
                        best_models_meta['rfc_model_id'] = row.ModelID
                    elif row.ModelID == 2:  # LR model
                        best_models_meta['lr_model_version_id'] = row.ModelVersionID
                        best_models_meta['lr_model_id'] = row.ModelID

                if None in best_models_meta.values():
                    logging.error("One or more model versions were not assigned correctly.")
                else:
                    logging.info('Fetching Training Data from DB')
                    unprocessed_training_data = db_instance.get_training_data()
                    if unprocessed_training_data:
                        logging.info('Initiating Retraining Process')
                        print('Initiating Retraining Process')

                        retraining_instance = ReTraining()
                        
                        best_models_and_data = retraining_instance.train_models(unprocessed_training_data, best_models_meta)
                        if None in best_models_and_data:
                            logging.error('One or more model retraining failed')
                            print('One or more model retraining failed')

                            db_instance.save_models_to_db(best_models_and_data, processing_status = 51) # status failed
                        else:
                            logging.info('Retraining Process Complete')
                            print('Retraining Process Complete')

                            logging.info("Saving Best Models to Database")
                            print("Saving Best Models to Database")

                            db_instance.save_models_to_db(best_models_and_data, processing_status = 1)# status processed
                            
                            logging.info("Succesfully Saved Models to Database")
                            print("Succesfully Saved Models to Database")
                    else:
                        logging.info("No training data available for retraining, Iteration Skipped")
                        print("No training data available for retraining, Iteration Skipped")
            else:
                logging.info('No Unprocessed Model Versions found, Iteration Skipped')
                print('No Unprocessed Model Versions found, Iteration Skipped')

        logging.info("Retraining Job Iteration Complete.")
        print("Retraining Job Iteration Complete.")

        logging.info('----------------------')
        print('----------------------')
    except Exception as e:
        logging.error('Retraining Retraining Iteration Failed with error: ' + str(e))
        logging.info('----------------------')

        print('Retraining Retraining Iteration Failed with error: ' + str(e))
        print('----------------------')


if __name__ == "__main__":

    logging.info("Insight job is running")
    print('Insight job is running')
    
    retraining_job()  
       
    # Interval time in seconds
    interval = 60 * 5  
    time.sleep(interval)