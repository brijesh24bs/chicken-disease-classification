from CNNClassifier.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from CNNClassifier.pipeline.base_model_pipeline import BaseModelPipeline
from CNNClassifier.pipeline.model_training_pipeline import ModelTrainingPipeline
from CNNClassifier.pipeline.evaluation_pipeline import EvaluationPipeline
# data_ingestion_pipeline = DataIngestionPipeline()
# data_ingestion_pipeline.run()

# base_model_pipeline = BaseModelPipeline()
# base_model_pipeline.run()

# model_training_pipeline = ModelTrainingPipeline()
# model_training_pipeline.run()

evaluation_pipeline = EvaluationPipeline()
evaluation_pipeline.run()



