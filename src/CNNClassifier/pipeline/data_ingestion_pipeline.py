from CNNClassifier.config.configuration_manager import ConfigurationManager
from CNNClassifier.components.data_ingestion import DataIngestion

class DataIngestionPipeline:
    def __int__(self):
        pass

    def run(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(data_ingestion_config)
            data_ingestion.download()
            data_ingestion.extract()
        except Exception as e:
            raise e


if __name__ == "__main__":
    try:
        pipeline = DataIngestionPipeline()
        pipeline.run()
    except  Exception as e:
        raise e