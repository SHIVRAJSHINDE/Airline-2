import sys
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.utils import save_object


from src.exception import CustomException
from src.logger import logging
import os


class transformClass:
    def initiate_data_transformation(self,data_train,data_test):

        try:
            train_df =  data_train
            test_df = data_test

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            target_column_name="Price"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]


            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            input_feature_test_df['Airline_Trujet'] = 0
            input_feature_test_df['Airline_MultiplecarriersPremiumeconomy'] = 0

            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            print(input_feature_train_df.columns)
            print(input_feature_test_df.columns)

            scaler = StandardScaler()
            input_feature_train_arr= scaler.fit_transform(input_feature_train_df)
            input_feature_test_arr= scaler.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            print(train_arr)
            return (
                train_arr,
                test_arr,
            )
        except Exception as e:
            raise CustomException(e,sys)

