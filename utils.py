import sys

import torch
import os
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import optim
from sklearn.svm import SVC

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# View the size of a list
def get_shape(lst):
    if not isinstance(lst, list):
        return ()
    return (len(lst),) + get_shape(lst[0])

# Merge two excel contents
def merge_excel():

    excel1 = pd.read_excel('F:/postgraduateProject/inHead/pythonProject/excel_data/testd.xlsx')
    excel2 = pd.read_excel('F:/postgraduateProject/inHead/pythonProject/excel_data/traind.xlsx')

    merged_excel = pd.concat([excel1, excel2])

    # Merging two tables, assuming they have the same column names
    # merged_excel = pd.merge(excel1, excel2)

    # Save the merged table as a new Excel file
    merged_excel.to_excel('F:/postgraduateProject/inHead/pythonProject/excel_data/merge.xlsx', index=False)

    return 0



from openpyxl import Workbook
def transfer_text_to_excel(text_path,output_path):

    data_folder = text_path

    subfolders = os.listdir(data_folder)

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


        print(get_shape(excel_data_merge))

    wb = Workbook()
    ws = wb.active

    for row in excel_data_merge:
        ws.append(row)

    # output_path = 'F:/postgraduateProject/inHead/pythonProject/excel_data/self_merge_text.xlsx'
    wb.save(output_path)

    print(f"The excel was created successfully, and it saved in {output_path}")

    return 0


def draw_loss_trainACC_testACC(loss_values,train_acc_values,test_acc_values):
    import matplotlib.pyplot as plt

    # Create three empty lists to hold loss, training accuracy and test accuracy
    loss_values = loss_values[4:]
    train_acc_values = train_acc_values
    test_acc_values = test_acc_values

    plt.figure(figsize=(10, 6))

    # Plotting loss curves
    plt.subplot(2, 1, 1)
    plt.plot(loss_values, label='Loss', marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting accuracy curves
    plt.subplot(2, 1, 2)
    plt.plot(train_acc_values, label='Train Acc', marker='o')
    plt.plot(test_acc_values, label='Test Acc', marker='o')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Adjusting the spacing between subgraphs
    plt.tight_layout()

    plt.show()
