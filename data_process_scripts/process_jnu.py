import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from itertools import islice
import numpy as np
from .sequence_aug import *


signal_size = 1024

# Three working conditions, with 'normal' labeled as 1 and 'anomaly' labeled as 0
WC1 = ["ib600.csv", "n600.csv", "ob600.csv", "tb600.csv"]
WC2 = ["ib800.csv", "n800.csv", "ob800.csv", "tb800.csv"]
WC3 = ["ib1000.csv", "n1000.csv", "ob1000.csv", "tb1000.csv"]

# Define normal and anomaly labels
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


# Generate Training Dataset and Testing Dataset
def get_files(root, condition_id):
    '''
    This function generates the final training and test sets based on the working condition ID.
    root: Location of the dataset
    condition_id: 1 for WC1, 2 for WC2, 3 for WC3
    '''
    data = []
    lab = []
    
    # Select the appropriate working condition based on the input condition_id
    if condition_id == 1:
        working_condition = WC1
    elif condition_id == 2:
        working_condition = WC2
    elif condition_id == 3:
        working_condition = WC3
    else:
        raise ValueError("Invalid condition ID. Choose 1, 2, or 3.")
    
    # Load the files for the selected working condition
    for filename in tqdm(working_condition):
        path = os.path.join(root, filename)
        label = normal_label if 'n' in filename else anomaly_label
        data_part, label_part = data_load(path, label)
        data += data_part
        lab += label_part

    return [data, lab]

def data_load(filename, label):
    '''
    This function is mainly used to generate test and training data.
    filename: Data location
    label: The label for the data (1 for normal, 0 for anomaly)
    '''
    fl = np.loadtxt(filename)
    fl = fl.reshape(-1, 1)
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

#--------------------------------------------------------------------------------------------------------------------
class JNU_Dataset(object):
    num_classes = 2  # Only two classes: normal and anomaly
    inputchannel = 1

    def __init__(self, data_dir, normlizetype, condition_id):
        self.data_dir = data_dir
        self.normlizetype = normlizetype
        self.condition_id = condition_id

    def data_prepare(self):
        # Get the data for the selected working condition
        list_data = get_files(self.data_dir, self.condition_id)
        
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
