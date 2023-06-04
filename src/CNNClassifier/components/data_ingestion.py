import os
import urllib.request as request
from zipfile import ZipFile
from CNNClassifier.logging import logger
from CNNClassifier.utils.common import get_size
from CNNClassifier.entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download(self):
        logger.info("Trying to download data from url: {}".format(self.config.source_url))
        if not os.path.exists(self.config.local_data_file):
            logger.info("Downloading started")
            filename, headers = request.urlretrieve(self.config.source_url, self.config.local_data_file)
            logger.info(f"Download completed. File size: {get_size(Path(self.config.local_data_file))}")
        else:
            logger.info("File already exists")

    def extract(self):
        logger.info("Trying to extract data from zip file")
        with ZipFile(file=self.config.local_data_file, mode="r") as zp:
            zp.extractall(self.config.unzip_dir)
        os.remove(self.config.local_data_file)
        logger.info("Extraction completed")




