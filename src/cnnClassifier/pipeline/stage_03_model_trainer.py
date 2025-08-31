from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_trainer import Training
from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_trainer import ModelTrainingPipeline



STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
            config = ConfigurationManager()
            training_config = config.get_training_config()
            training = Training(config = training_config)
            training.get_base_model()
            training.train_valid_generator()
            training.train()

if __name__ == "__main__" :
    try :
          logger.info(f"*******************")
          logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
          obj = ModelTrainingPipeline()
          obj.main()
          logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========X")
    except Exception as e:
         logger.exception(e)
         raise e


        