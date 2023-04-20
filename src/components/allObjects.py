


from src.components.data_ingestion import DataIngestionConfig
from src.components.data_ingestion import DataIngestion

from src.components.eda import EDA
from src.components.encoding import encodingclass


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



