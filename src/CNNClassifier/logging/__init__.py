import os
import sys
import logging

logdir = "logs"
log_filepath = os.path.join(logdir, "running_logs.log")

if not os.path.exists(logdir):
    os.makedirs(logdir, exist_ok=True)
with open(log_filepath, "w") as f:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("CnnClassifierLogger")