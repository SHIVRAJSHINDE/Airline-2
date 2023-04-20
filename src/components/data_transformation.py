import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformatoin:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        train_data = pd.read_csv("D:\\MachineLearningProjects\\PROJECT\\Airline-2\\artifacts\\train.csv")
        train_data.dropna(inplace=True)
        train_data['Date_of_Journey'] = pd.to_datetime(train_data['Date_of_Journey'], format="%d/%m/%Y")
        train_data['JourneyDay'] = train_data['Date_of_Journey'].dt.day
        train_data['JourneyMonth'] = train_data['Date_of_Journey'].dt.month
        train_data['JourneyYear'] = train_data['Date_of_Journey'].dt.year
        train_data.drop('Date_of_Journey', axis=1, inplace=True)

        train_data['Dep_Time'] = pd.to_datetime(train_data['Dep_Time'])
        train_data['Dep_Time_Hr'] = train_data['Dep_Time'].dt.hour
        train_data['Dep_Time_Min'] = train_data['Dep_Time'].dt.minute
        train_data.drop('Dep_Time', axis=1, inplace=True)

        train_data['Arrival_Time'] = pd.to_datetime(train_data['Arrival_Time'])
        train_data['Arrival_Time_Hour'] = train_data['Arrival_Time'].dt.hour
        train_data['Arrival_Time_Minute'] = train_data['Arrival_Time'].dt.minute
        train_data.drop('Arrival_Time', axis=1, inplace=True)

        duration = list(train_data['Duration'])
        duration
        hours = list(train_data['Duration'])

        for i in range(len(duration)):
            if "h" and "m" in duration[i]:
                a = duration[i].split(" ")[0]
                if "h" in a:
                    a = int(a.replace("h", "")) * 60
                elif "m" in a:
                    a = int(a.replace("m", ""))
            hours[i] = a

        train_data['DurationHr'] = train_data['Duration'].str.split(" ", expand=True).get(0)
        train_data['DurationMinute'] = train_data['Duration'].str.split(" ", expand=True).get(1)
        train_data['DurationMinute'] = train_data['DurationMinute'].fillna("0m")
        train_data['DurationMinute'].unique()

        DurationHr = list(train_data['DurationHr'])
        DurationMinute = list(train_data['DurationMinute'])
        DurationOnlyInMinutes = list()
        DurationMinute

        for i in range(len(DurationHr)):
            if "h" in DurationHr[i]:
                DurationHr[i] = int(DurationHr[i].replace("h", "")) * 60
            elif "m" in DurationHr[i]:
                DurationHr[i] = int(DurationHr[i].replace("m", ""))

        for i in range(len(DurationMinute)):
            if "h" in DurationMinute[i]:
                DurationMinute[i] = int(DurationMinute[i].replace("h", "")) * 60
            elif "m" in DurationMinute[i]:
                DurationMinute[i] = int(DurationMinute[i].replace("m", ""))

        DurationHrMin = DurationMinute + DurationHr

        for i in range(len(DurationHr)):
            DurationOnlyInMinutes.append(DurationHr[i] + DurationMinute[i])

        DurationOnlyInMinutes = pd.DataFrame(DurationOnlyInMinutes, columns=['DurationOnlyInMinutes'])
        DurationOnlyInMinutes

        train_data = pd.concat([train_data, DurationOnlyInMinutes], axis=1)

        train_data.drop(['Duration', 'DurationHr', 'DurationMinute'], axis=1, inplace=True)

        DurationOnlyInMinutes.head()

        # Since airline is nominal categorical data we will perform OneHotEncoding
        Airline = train_data[["Airline"]]
        Source = train_data[['Source']]
        Destination = train_data[['Destination']]
        Airline = pd.get_dummies(Airline, drop_first=True)
        Source = pd.get_dummies(Source, drop_first=True)
        Destination = pd.get_dummies(Destination, drop_first=True)
        train_data['Total_Stops'] = train_data['Total_Stops'].replace(
            {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4})

        data_train = pd.concat([train_data, Airline, Source, Destination], axis=1)

        data_train.drop(["Airline", "Source", "Destination", "Route", "Additional_Info"], axis=1, inplace=True)

        data_train.dropna(axis=0,inplace=True)


        print(data_train.head())
        print(data_train.columns)



if __name__ == "__main__":
    objdataTransformation = DataTransformatoin()
    objdataTransformation.get_data_transformer_object()

