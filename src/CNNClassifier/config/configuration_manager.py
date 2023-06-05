from CNNClassifier.constants import *
from CNNClassifier.utils.common import read_yaml, create_directories
from CNNClassifier.entity import DataIngestionConfig,BaseModelConfig,PrepareCallbackConfig,TrainingConfig
from pathlib import Path

import os
class ConfigurationManager:
    def __init__(self):
        self.config = read_yaml(CONFIG_FILE_PATH)
        self.params = read_yaml(PARAMS_FILE_PATH)

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

    def get_base_model_config(self) -> BaseModelConfig:
        config = self.config.base_model
        create_directories([config.root_dir])

        base_model_config = BaseModelConfig(
            root_dir=config.root_dir,
            base_model_path=config.base_model_path,
            updated_base_model_path=config.updated_base_model_path,
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES


        )
        return base_model_config

    def get_prepare_callbacks_config(self) -> PrepareCallbackConfig:
        config = self.config.prepare_callbacks
        create_directories([
            config.root_dir,
            config.tensorboard_root_log_dir,
            os.path.dirname(config.checkpoint_model_filepath)
        ])

        prepare_callbacks_config = PrepareCallbackConfig(
            root_dir=config.root_dir,
            tensorboard_root_log_dir=config.tensorboard_root_log_dir,
            checkpoint_model_filepath=config.checkpoint_model_filepath
        )
        return prepare_callbacks_config

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        base_model = self.config.base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "chicken-fecal-images")
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path = Path(training.trained_model_path),
            updated_base_model_path=Path(base_model.updated_base_model_path),
            training_data=Path(training_data),

            param_epochs=params.EPOCHS,
            param_batch_size=params.BATCH_SIZE,
            param_is_augmentation=params.AUGMENTATION,
            param_image_size=params.IMAGE_SIZE
        )

        return training_config

