from CNNClassifier.config.configuration_manager import ConfigurationManager
from CNNClassifier.components.evaluation import Evaluation
from CNNClassifier.logging import logger
class EvaluationPipeline:
    def __init__(self):
        pass

    def run(self):
        try:
            config = ConfigurationManager()
            val_config = config.get_validation_config()
            evaluation = Evaluation(val_config)
            evaluation.evaluation()
            evaluation.save_score()
        except Exception as e:
            logger.exception(e)


# if __name__ == '__main__':
#     try:
#         logger.info(f"*******************")
#         logger.info(f">>>>>> Evaluation Pipeline Started <<<<<<")
#         obj = EvaluationPipeline()
#         obj.run()
#         logger.info(f">>>>>> Evaluation Pipeline Completed <<<<<<\n\nx==========x")
#     except Exception as e:
#         logger.exception(e)
#         raise e
