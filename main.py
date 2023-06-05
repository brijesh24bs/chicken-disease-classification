from CNNClassifier.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from CNNClassifier.pipeline.base_model_pipeline import BaseModelPipeline
from CNNClassifier.pipeline.model_training_pipeline import ModelTrainng
# data_ingestion_pipeline = DataIngestionPipeline()
# data_ingestion_pipeline.run()

# base_model_pipeline = BaseModelPipeline()
# base_model_pipeline.run()

model_training_pipeline = ModelTrainng()
model_training_pipeline.run()

