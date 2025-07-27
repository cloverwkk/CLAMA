import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from .sequence_aug import *

signal_size = 1024

# label: 0 for normal, 1 for anomaly
normal_label = 0
anomaly_label = 1


class dataset(Dataset):

    def __init__(self, list_data, test=False, transform=None):
        self.test = test
        if self.test:
            self.seq_data = list_data['data'].tolist()
        else:
            self.seq_data = list_data['data'].tolist()
            self.labels = list_data['label'].tolist()
        if transform is None:
            self.transforms = Compose([
                Reshape()
            ])
        else:
            self.transforms = transform


    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, item):
        if self.test:
            seq = self.seq_data[item]
            seq = self.transforms(seq)
            return seq, item, item
        else:
            seq = self.seq_data[item]
            label = self.labels[item]
            seq = self.transforms(seq)
            return seq, label, item


# generate Training Dataset and Testing Dataset
def get_files(root, test=False):
    """
    This function is used to generate the final training set and test set.
    root: The location of the dataset
    """
    datasetname = os.listdir(root) # '1 - Three Baseline Conditions', etc.

    dataset1 = os.listdir(os.path.join(root, datasetname[0]))  # 'Three Baseline Conditions'
    dataset2 = os.listdir(os.path.join(root, datasetname[2]))  # 'Seven More Outer Race Fault Conditions'
    dataset3 = os.listdir(os.path.join(root, datasetname[3]))  # 'Seven Inner Race Fault Conditions'
    dataset4 = os.listdir(os.path.join(root, datasetname[1]))
    
    data_root1 = os.path.join(root, datasetname[0])  # Path of Three Baseline Conditions
    data_root2 = os.path.join(root, datasetname[2])  # Path of Seven More Outer Race Fault Conditions
    data_root3 = os.path.join(root, datasetname[3])  # Path of Seven Inner Race Fault Conditions
    data_root4 = os.path.join(root, datasetname[1])

    data = []
    lab = []

    for i in tqdm(range(len(dataset1))):
        path1 = os.path.join(data_root1, dataset1[i])
        data1, lab1 = data_load(path1, label=normal_label)  # Label for normal data is 1
        data += data1
        lab += lab1

    for i in tqdm(range(len(dataset2))):
        path2 = os.path.join(data_root2, dataset2[i])
        data1, lab1 = data_load(path2, label=anomaly_label)  # Label for anomaly data is 0
        data += data1
        lab += lab1

    for j in tqdm(range(len(dataset3))):
        path3 = os.path.join(data_root3, dataset3[j])
        data2, lab2 = data_load(path3, label=anomaly_label)  # Label for anomaly data is 0
        data += data2
        lab += lab2

    return [data, lab]

def data_load(filename, label):
    """
    This function is mainly used to generate test data and training data.
    filename: Data location
    label: 1 for normal, 0 for anomaly
    """
    if label == normal_label:
        fl = loadmat(filename)["bearing"][0][0][1]  # Take out the normal data
    else:
        fl = loadmat(filename)["bearing"][0][0][2]  # Take out the anomalous data

    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size
    return data, lab

def data_transforms(dataset_type="train", normlize_type="-1-1"):
    transforms = {
        'train': Compose([
            Reshape(),
            Normalize(normlize_type),
            Retype()
        ]),
        'val': Compose([
            Reshape(),
            Normalize(normlize_type),
            Retype()
        ])
    }
    return transforms[dataset_type]


# --------------------------------------------------------------------------------------------------------------------
class MFPT_Dataset(object):
    num_classes = 2  # Only two classes, 1: normal, 0: anomaly
    inputchannel = 1

    def __init__(self, data_dir, normlizetype):
        self.data_dir = data_dir
        self.normlizetype = normlizetype

    def data_prepare(self, test=False):
        list_data = get_files(self.data_dir, test=False)
        data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})

        # Split data into normal and anomaly sets
        normal_data = data_pd[data_pd["label"] == 0]
        anomaly_data = data_pd[data_pd["label"] == 1]

        if test:
            # Use half of the normal data and all the anomaly data for the test set
            normal_test_pd, _ = train_test_split(normal_data, test_size=0.5, random_state=40)
            test_pd = pd.concat([normal_test_pd, anomaly_data])
            test_dataset = dataset(list_data=test_pd, test=True, transform=None)
            return test_dataset
        else:
            # Split normal data into training and validation sets
            train_pd, normal_test_pd = train_test_split(normal_data, test_size=0.5, random_state=40)
            train_dataset = dataset(list_data=train_pd, transform=data_transforms('train', self.normlizetype))

            # Combine the remaining normal data with anomaly data for validation
            val_pd = pd.concat([normal_test_pd, anomaly_data])
            val_dataset = dataset(list_data=val_pd, transform=data_transforms('val', self.normlizetype))

            val_labels = val_pd["label"].values
            num_normal = sum(val_labels == 0)
            num_anomaly = sum(val_labels == 1)

            print(f"Validation dataset contains {num_normal} normal samples and {num_anomaly} anomaly samples.")

            return train_dataset, val_dataset
