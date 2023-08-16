import seqlog
import logging
from core.config import settings

seqlog.log_to_seq(
   server_url= settings.SEQ_SERVER,
   api_key= settings.SEQ_API_KEY,
   level=logging.INFO,
   batch_size=10,
   auto_flush_timeout=10, 
   override_root_logger=True,
   support_extra_properties=True
)