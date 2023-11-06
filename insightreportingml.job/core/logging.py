import seqlog
import logging
from core.config import settings

class SeqLogging:

      def __init__(self):
            seqlog.log_to_seq(
                  server_url= settings.SEQ_SERVER,
                  api_key= settings.SEQ_API_KEY,
                  level=logging.DEBUG,
                  batch_size=1,
                  auto_flush_timeout=10, 
                  override_root_logger=True,
                  support_extra_properties=True
            )
                  
            root_logger = logging.getLogger()
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            root_logger.addHandler(console_handler)