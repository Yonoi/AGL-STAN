import os
import numpy as np
from functools import reduce

import torch
import torch.utils.data as Data

from .normalization import StandardScaler, MinMax01Scaler
from .normalization import MinMax11Scaler, NScaler

# Testing print info.
from icecream import ic

class DatasetLoader(object):
    def __init__(self, args, num_workers=4):
        super(DatasetLoader, self).__init__()
        self._data_path = args.data_path
        self._adj_filename = args.adj_filename
        self._node_features_filename = args.node_features_filename
        
        # self._train_ratio = args.train_ratio
        # self._val_ratio = args.val_ratio
        self._batch_size = args.batch_size

        # self._norm = args.norm       
 
        self._num_nodes = args.num_nodes
        self._binary = args.binary
        self._window = args.window
        self._horizon = args.horizon

        self._num_workers = num_workers

        # for binary classification task
        self._pos_weights = None
        self._threshold = None

        self._read_data()
    
    def _read_data(self):
        A = np.load(os.path.join(self._data_path, self._adj_filename))
        X = np.load(os.path.join(self._data_path, self._node_features_filename))

        X = X.astype(np.float32)      

        # if self.binary is True, The downstream task is binary classification.
        # else, The downstream task is regression.
        if self._binary == 'true':
            X = np.int64(X > 0)
            self._norm = 'ns'
            X, scaler = self._normalize(X)
            self._threshold = np.sum(X, axis=(0, 1)) / reduce(lambda x, y : x * y, X.shape[:-1])
            self._pos_weights = torch.tensor((1 - self._threshold) / self._threshold)
            # maybe this threhold is better.
            self._threshold = np.array([0.5] * X.shape[-1])
            self._pos_weights = torch.tensor(len(self._threshold) * [1])
        else:
            X, scaler = self._normalize(X)

        self._adj = A
        self._data = X
        self._scaler = scaler 
    
    def _normalize(self, X):
        """
        1. StandardScaler : std
        2. MinMax01Scaler : max01
        3. MinMax11Scaler : max11
        4. NScaler : ns
        """
        if self._norm == 'std':
            mean = np.mean(X, axis=(0, 1), keepdims=True)
            std = np.std(X, axis=(0, 1), keepdims=True)
            scaler = StandardScaler(mean, std)
            X = scaler.transform(X)

        elif self._norm == 'max01':
            max = np.max(X, axis=(0, 1), keepdims=True)
            min = np.min(X, axis=(0, 1), keepdims=True)
            scaler = MinMax01Scaler(min, max)
            X = scaler.transform(X)

        elif self._norm == 'max11':
            max = np.max(X, axis=(0, 1), keepdims=True)
            min = np.min(X, axis=(0, 1), keepdims=True)
            scaler = MinMax11Scaler(min, max)
            X = scaler.transform(X)

        else:
            scaler = NScaler()
            X = scaler.transform(X)
        
        return X, scaler
        
    def _generate_task(self, data):
        indices = [
            (i, i + (self._window + self._horizon))
            for i in range(data.shape[0] - (self._window + self._horizon) + 1)
        ]

        # Generate observations
        features = []
        targets = []
        for i, j in indices:
            features.append(data[i : i + self._window, :, :])
            targets.append(data[i + self._window : j])
        
        # self._features = np.array(features)
        # self._targets = np.array(targets)
        return np.array(features), np.array(targets)

    def _train_test_split(self):
        """
        The build-in fucntion will be adapted to fit the Dynamic Graph data later
        """
        train_data = self._data[0:196] # 2015/01/01-2015/07/15
        val_data = self._data[196:212] # 2015/07/16-2015/07/31
        test_data_aug = self._data[212:242] # 2015/08/01-2015/08/31
        test_data_sep = self._data[242:273] # 2015/09/01-2015/09/31
        test_data_oct = self._data[273:304] # 2015/10/01-2015/10/31
        test_data_nov = self._data[304:334] # 2015/11/01-2015/11/30
        test_data_dec = self._data[334:365] # 2015/12/01-2015/12/31
        test_data = [
            test_data_aug, 
            test_data_sep,
            test_data_oct, 
            test_data_nov, 
            test_data_dec
        ]


        X_train, y_train = self._generate_task(train_data)    
        self._X_train, self._y_train =  (
            torch.Tensor(X_train),
            torch.Tensor(y_train)
        )

        X_val, y_val = self._generate_task(val_data)
        self._X_val, self._y_val = (
            torch.Tensor(X_val),
            torch.Tensor(y_val)
        )

        self._test_data_tensor = []
        for data in test_data:
            X_test, y_test = self._generate_task(data)
            self._test_data_tensor.append((
                torch.Tensor(X_test),
                torch.Tensor(y_test)
            ))

        train_loader = Data.DataLoader(
            dataset=Data.TensorDataset(self._X_train, self._y_train),
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self._num_workers
        )
        val_loader = Data.DataLoader(
            dataset=Data.TensorDataset(self._X_val, self._y_val),
            batch_size=self._batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self._num_workers
        )
        test_loaders = [
            Data.DataLoader(
            dataset=Data.TensorDataset(X_test, y_test),
            batch_size=self._batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self._num_workers)
            for X_test, y_test in self._test_data_tensor
        ]

        self._train_loader = train_loader
        self._val_loader = val_loader
        self._test_loaders = test_loaders

    def _data_info(self):
        print("Data Loaded Successfully!")
        if self._binary == 'true':
            print("The Task is Binary Classification.")
        else:
            print("The Task is Regression.")
        print()
        print("#" * 40 + "Data Info" + "#" * 40)

        print("The shape of adjacency matrix : {}".format(self._adj.shape))
        print("X_train shape : {}, y_train shape : {}".format(
            self._X_train.shape,
            self._y_train.shape
        ))
        print("X_val shape : {}, y_val shape : {}".format(
            self._X_val.shape,
            self._y_val.shape
        ))
        test_dataset_name = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for idx, data in enumerate(self._test_data_tensor):
            print("X_test_{} shape : {}, y_test_{} shape : {}".format(
                test_dataset_name[idx],
                data[0].shape,
                test_dataset_name[idx],
                data[1].shape
            ))
        print("The normalization method is {}.".format(self._norm))
        print("The scaler is : \n{}\n{}".format(
            self._scaler.get()[0],
            self._scaler.get()[1]    
        ))

        print()

    def get_dataset(self):
        self._train_test_split()
        self._data_info()
        return (
            (self._train_loader, self._val_loader, self._test_loaders), 
            self._adj, 
            self._scaler, 
            self._pos_weights,
            self._threshold
        )
