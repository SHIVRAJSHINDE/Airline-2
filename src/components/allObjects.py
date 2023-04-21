
from src.components.data_ingestion import DataIngestionConfig
from src.components.data_ingestion import DataIngestion

from src.components.eda import EDA
from src.components.encoding import encodingclass
from src.components.transformq import transformClass
from src.components.train import ModelTrainer
from src.components.train import ModelTrainerConfig


if __name__ == "__main__":
    createTrainTestFileObj = DataIngestion()
    train_path, test_path = createTrainTestFileObj.initiate_data_ingestion()

    edaObj = EDA()

    train_data = edaObj.edaOfTrainData(train_path)
    test_data = edaObj.edaOfTrainData(test_path)

    encoingObj1 = encodingclass(train_data)
    encoingObj2 = encodingclass(test_data)

    data_train = encoingObj1.trainDataencoding()
    data_test = encoingObj2.trainDataencoding()

    transformData = transformClass()
    train_arr, test_arr = transformData.initiate_data_transformation(data_train,data_test)

    modeltrainer = ModelTrainer()
    modeltrainer.initiate_model_trainer(train_arr, test_arr)




