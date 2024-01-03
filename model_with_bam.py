import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from torchvision.transforms import transforms
from torch.utils.data import Dataset, RandomSampler, DataLoader, random_split, ConcatDataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, r2_score
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from extractFeatures import get_text_from_excel_noramalization
import torch.nn.functional as F
from utils import get_shape, draw_loss_trainACC_testACC
import gradio as gr
import io
import base64
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix



train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])



class MyDataset(Dataset):
    def __init__(self, specs, labels):
        self.specs = specs
        self.labels = labels

    def __getitem__(self, index):
        spec, target = self.specs[index], self.labels[index]
        return spec.astype(np.float32), target

    def __len__(self):
        return len(self.specs)


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class BAMBlock(nn.Module):

    def __init__(self, channel, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

class NIR_CONV3(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding):
        super(NIR_CONV3, self).__init__()
        self.CONV1 = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.CONV2 = nn.Sequential(
            nn.Conv1d(output_channel, 83, 25, 1, 1),
            nn.BatchNorm1d(83),
            nn.ReLU(),
            nn.MaxPool1d(8)
        )
        self.CONV3 = nn.Sequential(
            nn.Conv1d(83, 32, 2, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(8)
        )

    def forward(self, x):
        x = self.CONV1(x)
        x = self.CONV2(x)
        x = self.CONV3(x)
        # x = self.CONV4(x)
        x = x.view(x.size(0), -1)  # 输出特征形状：64*1696  64：batch sizes

        return x



'''
self.feature_extractor(x):torch.Size([64, 2048])
features.unsqueeze(2).unsqueeze(3):torch.Size([64, 2048, 1, 1])
self.avg_pool(features):torch.Size([64, 2048, 1, 1])
pooled_features.view(pooled_features.size(0), -1):torch.Size([64, 2048])
'''
class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()
        self.feature_extractor = models.resnet50(pretrained=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity()
        self.attn = BAMBlock(2048, 16)


    def forward(self, x):
        # 检查输入维度
        if x.dim() == 3:
            x = x.unsqueeze(0)

        features = self.feature_extractor(x)
        features = features.unsqueeze(2).unsqueeze(3)
        pooled_features = self.avg_pool(features)
        pooled_features = self.attn(pooled_features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        # output = self.classifier(pooled_features)
        # print(f"FCN_output:{pooled_features}")
        return pooled_features


class DecisionLevelFusionRegression(nn.Module):
    def __init__(self):
        super(DecisionLevelFusionRegression, self).__init__()
        self.classifier1 = NIR_CONV3(1, 32, 19, 1, 0)
        self.classifier2 = FCN(num_classes=1)  # Regression: set num_classes=1

        self.fusion_layer = nn.Linear(2464, 1)  # Regression: output 1 value
        self.dropout = nn.Dropout(0.3)

    def forward(self, x1, x2):
        out1 = self.classifier1(x1)
        out2 = self.classifier2(x2)

        # Concatenate the outputs
        fused_output = torch.cat((out1, out2), dim=1)

        final_output = self.fusion_layer(fused_output)
        x = self.dropout(final_output)

        return x



class DecisionLevelFusion(nn.Module):
    def __init__(self, num_classes):
        super(DecisionLevelFusion, self).__init__()
        self.classifier1 = NIR_CONV3(1, 32, 19, 1, 0)
        self.classifier2 = FCN(num_classes=num_classes)
        self.fusion_layer = nn.Linear(2464, 7)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x1, x2):
        out1 = self.classifier1(x1)
        out2 = self.classifier2(x2)
        fused_output = torch.cat((out1, out2), dim=1)
        final_output = self.fusion_layer(fused_output)
        x = self.dropout(final_output)

        # To obtain AUC
        probs = torch.softmax(x, dim=1)

        return x, probs


def rf_img(path):
    # red data
    df = pd.read_excel(path)

    X = df.iloc[:, :-1]  # features
    y = df.iloc[:, -1]  # labels

    rf_classifier = RandomForestClassifier()

    # Train
    rf_classifier.fit(X, y)

    # Importance
    feature_importances = rf_classifier.feature_importances_

    # Correlate feature importance with feature name
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

    # Ranked according to importance of features
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Selection of thresholds, which can be adjusted on a case-by-case basis
    threshold = 0.008

    # Filtration characteristics
    selected_features = feature_importance_df[feature_importance_df['Importance'] > threshold]['Feature']

    # new datasets
    new_data = df[selected_features]
    # print(new_data)
    # Option to save the new data set to a file
    # new_data.to_excel("F:/postgraduateProject/inHead/pythonProject/excel_data/RF.xlsx", index=False)

    plt.figure(figsize=(12, 6))
    plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xticks(rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importance from Random Forest')


    img_path = "plot1.png"
    plt.savefig(img_path)
    # Wrap the image file into a gr.File object
    img_file = gr.File(img_path)
    plt.close()
    return img_file


def draw_loss_trainACC_testACC(train_loss_ep, train_acc_ep, test_acc_ep):
    # Assuming train_loss_ep, train_acc_ep, test_acc_ep are lists containing data

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plotting training loss
    plt.subplot(2, 1, 1)
    plt.plot(train_loss_ep, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting training and test accuracy
    plt.subplot(2, 1, 2)
    plt.plot(train_acc_ep, label='Training Accuracy')
    plt.plot(test_acc_ep, label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Save the plot as an image
    img_path = "plot.png"
    plt.savefig(img_path)
    plt.close()


    return img_path


def start(img_path, nir_path):

    # Defining the image dataset path
    data_path = img_path
    random_seed = 42
    merge_data = []
    # Load the dataset and segment it
    dataset = ImageFolder(data_path, transform=train_transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset_img, test_dataset_img = torch.utils.data.random_split(dataset,
                                                                        [train_size, test_size],
                                                                        generator=torch.Generator().manual_seed(
                                                                            random_seed))
    train_dataset_img.dataset.transform = train_transform
    test_dataset_img.dataset.transform = test_transform

    img_train_loader = DataLoader(train_dataset_img, batch_size=20, shuffle=True)
    img_test_loader = DataLoader(test_dataset_img, batch_size=20, shuffle=False)

    # NIRdata
    data_normal, data_normal_labels = get_text_from_excel_noramalization(nir_path)

    data_normal = data_normal[:, np.newaxis, :]
    text_data_dataset = MyDataset(data_normal, data_normal_labels)

    train_dataset_text, test_dataset_text = torch.utils.data.random_split(text_data_dataset,
                                                                          [train_size, test_size],
                                                                          generator=torch.Generator().manual_seed(
                                                                              random_seed))
    text_train_loader = DataLoader(train_dataset_text, batch_size=20, shuffle=True)
    text_test_loader = DataLoader(test_dataset_text, batch_size=20, shuffle=False)

    # labels
    label_names = dataset.classes

    # Defining the FCN model and optimiser
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    fusion_model = DecisionLevelFusion(num_classes=len(label_names)).to(device)

    # Defining the loss function and optimiser
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(fusion_model.parameters(), lr=0.003)
    sum_loss = 0
    train_sum_acc = 0.0
    test_sum = 0.0
    train_loss_ep = []

    train_acc_ep = []
    test_acc_ep = []
    final_precision = 0
    final_auc = 0
    final_f1 = 0
    final_recall = 0
    print("Loss,train_acc,test_acc")
    # 模型训练
    for epoch in range(20):

        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        train_acc = 0.0
        test_acc = 0.0
        train_loss = 0.0

        all_labels = []
        all_predict = []
        all_probs = []
        txt_temp = 0
        for (img_inputs, img_labels), (text_inputs, text_labels) in zip(img_train_loader, text_train_loader):
            text_inputs = Variable(text_inputs).type(torch.FloatTensor).to(device)  # batch x
            text_labels = Variable(text_labels).type(torch.LongTensor).to(device)  # batch y
            txt_temp = text_inputs
            img_inputs, img_labels = img_inputs.to(device), img_labels.to(device)

            output, grab = fusion_model(text_inputs, img_inputs)

            text_loss = criterion(output, text_labels)
            optimizer.zero_grad()
            # print(f"text_train_loss:{text_loss}")
            # img_loss.backward()
            text_loss.backward()
            optimizer.step()
            sum_loss += text_loss.item()
            _, predicted = torch.max(output.data,
                                     1)
            total += text_labels.size(0)
            correct += (predicted == text_labels).cpu().sum().data.numpy()

        train_loss = text_loss.item()
        train_acc = 100. * correct / total
        train_loss_ep.append(train_loss)
        train_acc_ep.append(train_acc)


        with torch.no_grad():

            correct = 0.0
            total = 0.0
            for (img_inputs, img_labels), (text_inputs, text_labels) in zip(img_test_loader, text_test_loader):
                fusion_model.eval()  # 不训练
                text_inputs = Variable(text_inputs).type(torch.FloatTensor).to(device)  # batch x
                text_labels = Variable(text_labels).type(torch.LongTensor).to(device)  # batch y
                img_inputs, img_labels = img_inputs.to(device), img_labels.to(device)

                outputs, probs = fusion_model(text_inputs, img_inputs)
                _, predicted = torch.max(outputs.data,
                                         1)

                total += text_labels.size(0)
                correct += (predicted == text_labels).sum().cpu()


                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(text_labels.cpu().numpy())

            average = 'macro'
            y_true = np.array(all_labels)
            y_pred = np.argmax(np.array(all_probs), axis=1)
            # Calculate the confusion matrix
            conf_mat = confusion_matrix(y_true, y_pred)

            # Visual Confusion Matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
            plt.title('Test Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()

            y_scores = np.array(all_probs)

            precision = precision_score(y_true, np.argmax(y_scores, axis=1), average=average)
            recall = recall_score(y_true, np.argmax(y_scores, axis=1), average=average)
            f1 = f1_score(y_true, np.argmax(y_scores, axis=1), average=average)
            auc_score = roc_auc_score(y_true, all_probs, multi_class='ovr', average='macro')

            final_precision = precision
            final_f1 = f1
            final_auc = auc_score
            final_recall = recall


            acc = 100. * correct / total
            test_acc_ep.append(acc)

            print("train_loss= {:.4f}".format(train_loss), "train_acc= {:.4f}".format(train_acc), "test_acc= {:.4f}".format(acc))
    message = f"Precision: {final_precision:.4f}, Recall: {final_recall:.4f}, F1-Score: {final_f1:.4f}, AUC: {final_auc:.4f},Accuracy:{test_acc_ep[-1]}"

    img = draw_loss_trainACC_testACC(train_loss_ep, train_acc_ep, test_acc_ep)





if __name__ == '__main__':

    nir_data_path = "F:/postgraduateProject/inHead/test/xjtlu_lxh-main/excel_data/self_merge_text_operation.xlsx"

    img_data_path = 'F:/postgraduateProject/inHead/test/xjtlu_lxh-main/waveletImage/'
    start(img_data_path,nir_data_path)
