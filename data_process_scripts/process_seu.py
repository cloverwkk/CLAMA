import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from itertools import islice
from .sequence_aug import *

signal_size = 1024

# Data names of 5 bearing fault types under two working conditions
Bdata1 = ["health_20_0.csv", "ball_20_0.csv", "comb_20_0.csv", "inner_20_0.csv", "outer_20_0.csv"]
Bdata2 = ["health_30_2.csv", "ball_30_2.csv", "comb_30_2.csv", "inner_30_2.csv", "outer_30_2.csv"]

# Data names of 5 gear fault types under two working conditions
Gdata1 = ["Health_20_0.csv", "Chipped_20_0.csv", "Miss_20_0.csv", "Root_20_0.csv", "Surface_20_0.csv"]
Gdata2 = ["Health_30_2.csv", "Chipped_30_2.csv", "Miss_30_2.csv", "Root_30_2.csv", "Surface_30_2.csv"]

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

# Generate training and testing datasets based on the selected dataset
def get_files(root, dataset_type="Bdata1"):
    '''
    This function is used to generate the final training set and test set.
    root: The location of the dataset
    dataset_type: The type of dataset (Bdata1, Bdata2, Gdata1, Gdata2)
    '''
    dataset_map = {
        "Bdata1": Bdata1,
        "Bdata2": Bdata2,
        "Gdata1": Gdata1,
        "Gdata2": Gdata2
    }
    data_list = dataset_map[dataset_type]

    if dataset_type in ["Bdata1", "Bdata2"]:
        root_path = os.path.join(root, 'bearingset')  # Get the path based on dataset type
    else:
        root_path = os.path.join(root, 'gearset')  # Get the path based on dataset type

    data = []
    lab = []

    for i in tqdm(range(len(data_list))):
        path = os.path.join(root_path, data_list[i])
        label = normal_label if "health" in data_list[i].lower() else anomaly_label
        data1, lab1 = data_load(path, dataname=data_list[i], label=label)
        data += data1
        lab += lab1

    return [data, lab]

# Data loading function
def data_load(filename, dataname, label):
    '''
    This function is mainly used to generate test data and training data.
    filename: Data location
    '''
    with open(filename, "r", encoding='gb18030', errors='ignore') as f:
        fl = []
        for line in islice(f, 16, None):  # Skip the first 16 lines
            line = line.rstrip()
            raw_word = line.split(",", 8) if dataname == "ball_20_0.csv" else line.split("\t", 8)[0].split(',')
            fl.append(eval(raw_word[1]))  # Take a vibration signal in the x direction as input
        fl = np.array(fl).reshape(-1, 1)

    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size

    return data, lab

# Transformations
def data_transforms(dataset_type="train", normalize_type="-1-1"):
    transforms = {
        'train': Compose([
            Reshape(),
            Normalize(normalize_type),
            Retype()
        ]),
        'val': Compose([
            Reshape(),
            Normalize(normalize_type),
            Retype()
        ])
    }
    return transforms[dataset_type]


#--------------------------------------------------------------------------------------------------------------------
class SEU_Dataset(object):
    num_classes = 2
    inputchannel = 1

    def __init__(self, data_dir, normlizetype, dataset_type):
        self.data_dir = data_dir
        self.normlizetype = normlizetype
        self.dataset_type = dataset_type

    def data_prepare(self):
        # Get all data
        list_data = get_files(self.data_dir, self.dataset_type)
        
        # Create a DataFrame with data and labels
        data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
        
        # Separate normal data (label = 1) and anomaly data (label = 0)
        normal_data = data_pd[data_pd['label'] == 0]
        anomaly_data = data_pd[data_pd['label'] == 1]

        # Split normal data into 50% training and 50% testing
        train_normal, test_normal = train_test_split(normal_data, test_size=0.5, random_state=40)
        
        # Test set consists of the remaining normal data and all anomaly data
        test_data = pd.concat([test_normal, anomaly_data])

        # Print the number of samples in test_normal (normal data in test set) and anomaly_data
        print(f"Number of normal samples in test set: {len(test_normal)}")
        print(f"Number of anomaly samples in test set: {len(anomaly_data)}")

        # Create training and testing datasets
        train_dataset = dataset(list_data=train_normal, transform=data_transforms('train', self.normlizetype))
        test_dataset = dataset(list_data=test_data, transform=data_transforms('val', self.normlizetype))

        return train_dataset, test_dataset
