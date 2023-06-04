from CNNClassifier.config.configuration_manager import ConfigurationManager
from CNNClassifier.entity import BaseModelConfig
from CNNClassifier.components.base_model import BaseModel

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
