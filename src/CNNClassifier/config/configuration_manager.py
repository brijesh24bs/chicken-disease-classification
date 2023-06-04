from CNNClassifier.constants import *
from CNNClassifier.utils.common import read_yaml, create_directories
from CNNClassifier.entity import DataIngestionConfig
from pathlib import Path

import os
class ConfigurationManager:
    def __init__(self):
        self.config = read_yaml(Path(CONFIG_FILE_PATH))
        self.params = read_yaml(Path(PARAMS_FILE_PATH))

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        return data_ingestion_config
