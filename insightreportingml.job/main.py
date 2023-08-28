import sys
import logging
from core.logging import initialise_seq
from scripts.retraining import retraining

initialise_seq()

try:
    logging.info("Job Retraining started")

    # Read training data
    data = retraining.read_from_excel()

    # Generate models from training data
    [acc, roc_auc, lr_tfidf, tfidf_vectorizer, rfc_model,
        preprocessing] = retraining.generate_models(data)

    logging.info(
        'Training models generated successfully-{acc}:{roc_auc}', acc=str(round(acc,4)), roc_auc=str(round(roc_auc, 4)))

    # Serilized training model to disk
    retraining.save_models('3', lr_tfidf, tfidf_vectorizer, rfc_model,
                           preprocessing)

    sys.exit(0)

except Exception as e:
    logging.error('Error while generating training models: ' + str(e))
    sys.exit(1)
