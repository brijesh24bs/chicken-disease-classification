from CNNClassifier.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from CNNClassifier.pipeline.base_model_pipeline import BaseModelPipeline
# data_ingestion_pipeline = DataIngestionPipeline()
# data_ingestion_pipeline.run()

base_model_pipeline = BaseModelPipeline()
base_model_pipeline.run()
