from CNNClassifier.config.configuration_manager import ConfigurationManager
from CNNClassifier.components.base_model import BaseModel
from CNNClassifier.logging import logger

class BaseModelPipeline:
    def __init__(self):
        pass

    def run(self):
        try:
            config = ConfigurationManager()
            base_model_config = config.get_base_model_config()
            base_model = BaseModel(base_model_config)
            base_model.get_base_model()
            base_model.update_base_model()
        except Exception as e:
            raise e



if __name__ == "__main__":
    try:
        pipeline = BaseModelPipeline()
        pipeline.run()
    except Exception as e:
        raise e
