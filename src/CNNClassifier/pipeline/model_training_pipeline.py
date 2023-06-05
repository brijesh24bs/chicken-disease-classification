from CNNClassifier.config.configuration_manager import ConfigurationManager
from CNNClassifier.entity import PrepareCallbackConfig, TrainingConfig

from CNNClassifier.components.prepare_callback import PrepareCallback
from CNNClassifier.components.model_training import ModelTraining



class ModelTrainingPipeline:
    def __init__(self):
        pass

    def run(self):
        try:
            config = ConfigurationManager()
            prepare_callback_config = config.get_prepare_callbacks_config()
            prepare_callback = PrepareCallback(prepare_callback_config)
            callback_list = prepare_callback.get_tb_ckpt_callbacks()

            training_config = config.get_training_config()
            model_trainer = ModelTraining(training_config)
            model_trainer.get_base_model()
            model_trainer.train_valid_generator()
            model_trainer.train(callback_list=callback_list)

        except Exception as e:
            raise e


if __name__ == "__main__":
    try:
        model_training_pipeline = ModelTrainingPipeline()
        model_training_pipeline.run()
    except Exception as e:
        raise e