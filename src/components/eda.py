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
from src.components.encoding import encodingclass


class EDA:

    def edaOfTrainData(self,path):
        try:
            self.df = pd.read_csv(path)
            self.df.dropna(inplace=True)
            self.df['Date_of_Journey'] = pd.to_datetime(self.df['Date_of_Journey'], format="%d/%m/%Y")
            self.df['JourneyDay'] = self.df['Date_of_Journey'].dt.day
            self.df['JourneyMonth'] = self.df['Date_of_Journey'].dt.month
            self.df['JourneyYear'] = self.df['Date_of_Journey'].dt.year

            self.df['Dep_Time'] = pd.to_datetime(self.df['Dep_Time'])
            self.df['Dep_Time_Hr'] = self.df['Dep_Time'].dt.hour
            self.df['Dep_Time_Min'] = self.df['Dep_Time'].dt.minute

            self.df['Arrival_Time'] = pd.to_datetime(self.df['Arrival_Time'])
            self.df['Arr_Time_Hr'] = self.df['Arrival_Time'].dt.hour
            self.df['Arr_Time_Min'] = self.df['Arrival_Time'].dt.minute

            duration = list(self.df['Duration'])

            hours = list(self.df['Duration'])

            for i in range(len(duration)):
                if "h" and "m" in duration[i]:
                    a = duration[i].split(" ")[0]
                    if "h" in a:
                        a = int(a.replace("h", "")) * 60
                    elif "m" in a:
                        a = int(a.replace("m", ""))
                hours[i] = a

            self.df['DurationHr'] = self.df['Duration'].str.split(" ", expand=True).get(0)
            self.df['DurationMinute'] = self.df['Duration'].str.split(" ", expand=True).get(1)
            self.df['DurationMinute'] = self.df['DurationMinute'].fillna("0m")

            DurationHr = list(self.df['DurationHr'])
            DurationMinute = list(self.df['DurationMinute'])
            DurationOnlyInMinutes = list()


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

            self.df = pd.concat([self.df, DurationOnlyInMinutes], axis=1)
            self.df.drop(['Date_of_Journey', 'Duration', 'DurationHr', 'DurationMinute', 'Dep_Time', 'Arrival_Time',
                                  'Additional_Info'], axis=1, inplace=True)

            self.df.dropna(inplace=True)
            print("witout Encoding")
            print(self.df)
            return self.df

        except Exception as e:
            raise CustomException(e,sys)


