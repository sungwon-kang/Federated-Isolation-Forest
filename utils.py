import math
import random
import numpy as np
import pandas as pd

class util:

    def __init__(self, seed, alpha=2, isiid=False, n_clients=10):
        self.seed = seed
        self.n_clients = n_clients
        self.isiid=isiid
        self.alpha=alpha
        self.nclasses = -1
        random.seed(seed)
        np.random.seed(seed)

    def getdata(self,filename, outlier, test_size=0.2):
        df = pd.read_csv(f'./data/{filename}.csv')
        data = df.values

        classes = np.unique(data[:, -1])
        self.nclasses = len(classes)

        classes_subset = self.split_class_data(data)
        train, test = self.split_train_test(classes_subset, test_size)

        train_subset = self.split_class_data(train)
        if self.isiid:
            train = self.assign_data_client_IIDsetting(train_subset, outlier)
        else:
            train = self.assign_data_client_nonIIDsetting(train_subset, outlier, self.alpha)

        label = self.transform_binary_class(test, outlier)
        test = test[:, :-1]

        return train, test, label

    def transform_binary_class(self, data, outlier):
        label = [1 if e[-1] == outlier else 0 for e in data]
        return label

    def split_class_data(self, data):

        classes_lst = []
        for i in range(self.nclasses):
            classes_lst.append(data[data[:, -1] == i, :])

        return classes_lst

    def split_train_test(self, classes_subset, test_size):

        train = []
        test = []
        train_size = 1.0 - test_size
        for subset in classes_subset:
            offset = math.floor(subset.shape[0] * train_size)
            np.random.shuffle(subset)

            train += subset[0:offset].tolist()
            test += subset[offset:].tolist()

        train = np.array(train)
        test = np.array(test)

        return train, test

    def assign_data_client_IIDsetting(self, train, outlier):
        clients = [[] for _ in range(self.n_clients)]

        for i in range(self.nclasses):
            if i == outlier: continue
            subset=train[i]
            offset = int(subset.shape[0] / self.n_clients)

            start = 0
            end = offset
            for j in range(self.n_clients):
                clients[j] += subset[start:end].tolist()
                start = end
                end = end + offset

            clients[self.n_clients - 1] += subset[start:].tolist()

        for i, client in enumerate(clients):
            client = np.array(client)
            client = np.delete(client, client.shape[1] - 1, axis=1)
            np.random.shuffle(client)
            clients[i] = client

        return clients

    def assign_data_client_nonIIDsetting(self, train, outlier, alpha):

        train_client = []
        for i in range(self.nclasses):
            if i == outlier: continue

            subset = train[i]
            offset = math.floor(subset.shape[0] / alpha)

            start = 0
            end = offset
            for _ in range(alpha - 1):
                train_client.append(subset[start:end])
                start = end
                end = end + offset
            train_client.append(subset[start:])

        random.shuffle(train_client)
        clients = [[] for _ in range(self.n_clients)]
        data_index = np.random.permutation(len(train_client))
        for i in data_index:
            j = i % self.n_clients
            clients[j] += train_client[data_index[i]].tolist()

        final_train = []
        for c in clients:
            c = np.array(c)
            c = c[:, :-1]
            final_train.append(c)

        return final_train

