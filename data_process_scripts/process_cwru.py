import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from .sequence_aug import *


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


signal_size = 1024

normalname1 = ["97.mat"] # 1797rpm
normalname2 = ["98.mat"] # 1772rpm
normalname3 = ["99.mat"] # 1750rpm
normalname4 = ["100.mat"] # 1730rpm

abnomalyname1 = ["105.mat", "118.mat", "130.mat", "169.mat", "185.mat", "197.mat", "209.mat", "222.mat",
             "234.mat"]  # 1797rpm
abnomalyname2 = ["106.mat", "119.mat", "131.mat", "170.mat", "186.mat", "198.mat", "210.mat", "223.mat",
             "235.mat"]  # 1772rpm
abnomalyname3 = ["107.mat", "120.mat", "132.mat", "171.mat", "187.mat", "199.mat", "211.mat", "224.mat",
             "236.mat"]  # 1750rpm
abnomalyname4 = ["108.mat", "121.mat", "133.mat", "172.mat", "188.mat", "200.mat", "212.mat", "225.mat",
             "237.mat"]  # 1730rpm

# 标签
normal_label = 0 
anomaly_label = 1  
axis = ["_DE_time", "_FE_time", "_BA_time"]


def get_files(root, normalname, abnomalyname, test=False):
    data_root = root

    # 添加子路径
    normal_path = os.path.join(data_root, "Normal Baseline Data", normalname[0])
    data, lab = data_load(normal_path, axisname=normalname[0], label=normal_label)

    for i in tqdm(range(len(abnomalyname))):
        fault_path = os.path.join(data_root, "12k Drive End Bearing Fault Data", abnomalyname[i])
        data1, lab1 = data_load(fault_path, axisname=abnomalyname[i], label=anomaly_label)
        data += data1
        lab += lab1

    return [data, lab]


def data_load(filename, axisname, label):
    datanumber = axisname.split(".")[0]
    if int(datanumber) < 100:
        realaxis = "X0" + datanumber + axis[0]
    else:
        realaxis = "X" + datanumber + axis[0]

    fl = loadmat(filename)[realaxis]
    data = []
    lab = []
    start, end = 0, signal_size
    
    while end <= fl.shape[0]:
        segment = fl[start:end]

        if segment.shape[0] == signal_size:
            data.append(segment.reshape(1, -1))  
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


class CWRU_Dataset(object):
    num_classes = 2
    inputchannel = 1

    def __init__(self, data_dir, normlizetype, data_index=1):
        self.data_dir = data_dir
        self.normlizetype = normlizetype

        self.normalname = globals()[f"normalname{data_index}"]
        self.abnomalyname = globals()[f"abnomalyname{data_index}"]

    def data_prepare(self, test=False):
        list_data = get_files(self.data_dir, self.normalname, self.abnomalyname, test)
        data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})

        normal_data = data_pd[data_pd["label"] == 0]
        anomaly_data = data_pd[data_pd["label"] == 1]

        if test:
            normal_test_pd, _ = train_test_split(normal_data, test_size=0.5, random_state=40)
            test_pd = pd.concat([normal_test_pd, anomaly_data])
            test_dataset = dataset(list_data=test_pd, test=True, transform=None)
            return test_dataset
        else:
            train_pd, normal_test_pd = train_test_split(normal_data, test_size=0.5, random_state=40)
            train_dataset = dataset(list_data=train_pd, transform=data_transforms('train', self.normlizetype))

            test_pd = pd.concat([normal_test_pd, anomaly_data])
            val_dataset = dataset(list_data=test_pd, transform=data_transforms('val', self.normlizetype))

            val_labels = test_pd["label"].values
            num_normal = sum(val_labels == 0)
            num_anomaly = sum(val_labels == 1)

            print(f"Validation dataset contains {num_normal} normal samples and {num_anomaly} anomaly samples.")

            return train_dataset, val_dataset