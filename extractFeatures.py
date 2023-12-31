import sys
import pandas as pd
import torch
import os
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import optim
from sklearn.svm import SVC
from openpyxl import Workbook
from sklearn.preprocessing import scale, MinMaxScaler, Normalizer, StandardScaler

from utils import *

def get_shape(lst):
    if not isinstance(lst, list):
        return ()
    return (len(lst),) + get_shape(lst[0])

def get_picFeature(path):

    model_path = path

    model_pic = torch.load(model_path)

    pic_data = []

    for row in range(len(model_pic[0])):
        for i in range(len(model_pic[0][row])):
            pic_data.append(model_pic[0][row][i].tolist())

    return pic_data

def get_text_from_excel_origin():

    data_path = "F:/postgraduateProject/inHead/pythonProject/excel_data/self_merge_text_operation.xlsx"
    # data_path = "F:/postgraduateProject/inHead/pythonProject/excel_data/RF.xlsx"
    data_origin = pd.read_excel(data_path)
    labels = np.array(data_origin.iloc[:, -1].values)
    data = np.array(data_origin.iloc[:, :-3])

    return data,labels

def get_text_from_excel_noramalization(path):

    data_path = path
    data_origin = pd.read_excel(data_path)
    labels = np.array(data_origin.iloc[:, -1].values)

    data = np.array(data_origin.iloc[:, :-2])
    row,col = data.shape

    data_d2 = np.ones((row,col-2))
    for i in range(len(data)):

        data_d2[i] = np.diff(np.diff(data[i]))

    data_normal = scale(data_d2)

    return data_normal,labels

def get_textFeature(path):

    data_folder = path
    subfolders = os.listdir(data_folder)

    sec_data = []
    labels = []
    excel_data_merge = []

    for subfolder in subfolders:
        subfolder_path = os.path.join(data_folder, subfolder)
        if os.path.isdir(subfolder_path):
            for txt_file in os.listdir(subfolder_path):
                if txt_file.endswith(".txt"):
                    txt_file_path = os.path.join(subfolder_path, txt_file)

                    with open(txt_file_path, 'r') as file:
                        data = file.readlines()
                    data = [line.strip().split(',') for line in data]

                    x = [float(row[0]) for row in data]
                    y = [float(row[1]) for row in data]
                    y.append(subfolder)
                    if excel_data_merge == "":
                        excel_data_merge.append(x)
                    else:
                        excel_data_merge.append(y)

                    labels.append(subfolder)

    return sec_data,labels,excel_data_merge

def get_useful_features(path):

    data_folder = path

    subfolders = os.listdir(data_folder)

    for subfolder in subfolders:
        subfolder_path = os.path.join(data_folder, subfolder)
        if os.path.isdir(subfolder_path):
            for txt_file in os.listdir(subfolder_path):
                if txt_file.endswith(".txt"):
                    txt_file_path = os.path.join(subfolder_path, txt_file)
                    with open(txt_file_path,'r+') as file:
                        lines = file.readlines()
                        file.seek(0)
                        file.truncate()
                        count = 0
                        for line in lines:
                            columns = line.split(',')
                            if len(columns) >= 1 and (956 <= float(columns[0]) <= 1215) or (
                                    1592 <= float(columns[0]) <= 1709):
                                count += 1
                                file.write(line)

    return 0

def combine_features(pic_path,text_path):
    features = []
    pic_features = torch.tensor(get_picFeature(pic_path))
    text_features,labels = get_textFeature(text_path)
    text_features = torch.tensor(text_features)

    if len(pic_features) == len(text_features):
        for i in range(len(pic_features)):
            features.append(torch.cat((pic_features[i], text_features[i]), dim=0))
    else:
        print("the length of pic_features and text_features are not equal")
        sys.exit(1)
    return features,labels

def get_data():
    pic_path = 'F:/postgraduateProject/inHead/pythonProject/train_features_labels.pt'
    text_path = "F:/postgraduateProject/inHead/pythonProject/血迹"
    features, labels = combine_features(pic_path, text_path)
    return features, labels
