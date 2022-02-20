from typing import Tuple
from PIL import Image
import os

import torch
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import LabelEncoder


class DataframePreprocessing:
    """
    Class, which splits csv file to train and test and encode labels
    """

    def __init__(self, path_to_csv: str, label_to_encode: str) -> None:
        df = pd.read_csv(path_to_csv)

        self.le = LabelEncoder()
        self.le.fit(df[label_to_encode])

        df['labels'] = self.le.transform(df[label_to_encode])

        train_df = df[df['is_valid'] == False]
        val_df = df[df['is_valid'] == True]

        self.train_img_paths = train_df['path'].to_numpy()
        self.train_img_labels = train_df['labels'].to_numpy()

        self.val_img_paths = val_df['path'].to_numpy()
        self.val_img_labels = val_df['labels'].to_numpy()

    def get_train_data(self) -> Tuple[np.array, np.array]:
        return self.train_img_paths, self.train_img_labels
        
    def get_valid_data(self) -> Tuple[np.array, np.array]:
        return self.val_img_paths, self.val_img_labels


class DatasetProcessing(Dataset):
    """
    Image proccesing module for batch generator
    """

    def __init__(self, data_path, img_paths, img_labels, transform=None):
        self.img_path = data_path
        self.transform = transform

        self.img_filename = img_paths
        self.labels = img_labels

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label = torch.from_numpy(np.array(self.labels[index])).type(torch.LongTensor)
        return img, label

    def __len__(self):
        return len(self.img_filename)