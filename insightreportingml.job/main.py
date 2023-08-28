import logging
import core.logging
from scripts.retraining import retraining

# Read training data
data = retraining.read_from_excel()

# Generate models from training data
[acc, roc_auc, lr_tfidf, tfidf_vectorizer, rfc_model,
    preprocessing] = retraining.generate_models(data)

print(acc, roc_auc)

# Serilized training model to disk
retraining.save_models('3', lr_tfidf, tfidf_vectorizer, rfc_model,
                       preprocessing)
