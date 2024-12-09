import numpy as np
import pandas as pd
import math

import sys
global split_count
split_count=0

# iTree의 n개의 노드를 가질 때, 평균 path length 추정치
def c(size):
    if size > 2:
        return 2 * (np.log(size-1)+0.5772156649) - 2*(size-1)/size
    if size == 2:
        return 1
    return 0
    
# external 노드
class LeafNode:
    def __init__(self, size):
        self.size = size
        # self.data = data
        
    def setSize(self, size):
        self.size=size
        
    # def setData(self, data):
    #     self.data=data
        
    def getSize(self):
        return self.size
    
    # def getData(self):
    #     return self.data
    
    # def PRINT(self):
    #     print(f"height: {self.__height__}, data: {print_count(self.data)} {self.size}")

    
# internal 노드
class DecisionNode:
    def __init__(self, left, right, splitAtt, splitVal):
        self.left = left
        self.right = right
        self.splitAtt = splitAtt
        self.splitVal = splitVal
        
    def getAtt(self):
        return self.splitAtt, self.splitVal

# iTree
class IsolationTree:

    def __init__(self, height, height_limit):
        self.height = height  # 현재의 트리 높이
        self.height_limit = height_limit  # 트리의 높이 한계
        self.root = None
        self.total_leafsize=2**height_limit
        
        
    def setheightlimit(self, h):
        self.height_limit = h
    
    def getRoot(self):
        return self.root
    
    def fit(self, X: np.ndarray, leaf_size):

        # 논문에 있는 Original 코드
        if self.height >= self.height_limit or X.shape[0] <= 2:
            self.root = LeafNode(X.shape[0] + leaf_size)
            return self.root

        num_features = X.shape[1]  # 속성의 갯수(열)을 저장
        splitAtt = np.random.randint(0, num_features)  # 0 ~ 속성의 갯수를 랜덤으로 정함
        splitVal = np.random.uniform(min(X[:, splitAtt]), max(X[:, splitAtt]))  # 선택된 속성의 min max 사이의 값 저장

        X_left = X[X[:, splitAtt] < splitVal]
        X_right = X[X[:, splitAtt] >= splitVal]

        ratio_left = X_left.shape[0] / X.shape[0]

        leaf_left = leaf_size * ratio_left
        leaf_right = leaf_size - leaf_left

        global split_count
        split_count += 1

        left = IsolationTree(self.height + 1, self.height_limit)
        right = IsolationTree(self.height + 1, self.height_limit)

        # 재귀 호출하여 진행
        left.fit(X_left, leaf_left)
        right.fit(X_right, leaf_right)
        self.root = DecisionNode(left.root, right.root, splitAtt, splitVal)

        return self.root

    
    def grow(self, X):
        if isinstance(self.root, LeafNode) and (self.height >= self.height_limit or X.shape[0] <= 2): 
            leaf_size = self.root.getSize()
            self.root.setSize(leaf_size + X.shape[0])
            return self.root

        left = IsolationTree(self.height + 1, self.height_limit)
        right = IsolationTree(self.height + 1, self.height_limit) 
        
        global split_count
        split_count+=1
        
        if isinstance(self.root, DecisionNode):
           
            left.root = self.root.left
            right.root = self.root.right
            
            X_left = X[X[:, self.root.splitAtt] < self.root.splitVal]
            X_right = X[X[:, self.root.splitAtt] >= self.root.splitVal]  
            
            self.root.left = left.grow(X_left)
            self.root.right = right.grow(X_right)
            
        elif isinstance(self.root, LeafNode):
            
            num_features = X.shape[1]                     
            splitAtt = np.random.randint(0, num_features)  
            splitVal = np.random.uniform(min(X[:, splitAtt]), max(X[:, splitAtt])) 
            
            X_left = X[X[:, splitAtt] < splitVal]
            X_right = X[X[:, splitAtt] >= splitVal]  
            
            leaf_size = self.root.getSize()
            ratio_left = X_left.shape[0] / X.shape[0]
            
            leaf_left = leaf_size * ratio_left
            leaf_right = leaf_size - leaf_left
            
            left.fit(X_left, leaf_left)
            right.fit(X_right, leaf_right)

            self.root = DecisionNode(left.root, right.root, splitAtt, splitVal)
        
        return self.root
    
    def supply(self, root, X):
        if isinstance(root, DecisionNode):
           
            X_left = X[X[:, root.splitAtt] < root.splitVal]
            X_right = X[X[:, root.splitAtt] >= root.splitVal]  
            
            global split_count
            split_count+=1
            
            left=root.left
            right=root.right
            
            return self.supply(left, X_left) + self.supply(right, X_right)
                        
        elif isinstance(root, LeafNode):
            size = root.getSize()
            size = size + X.shape[0]
            root.setSize(size)
            
            return size
        
        return 0
    
    def normalize(self, root, sample_size, total_size):
        
        if isinstance(root, DecisionNode):
            
            left=self.normalize(root.left, sample_size, total_size)
            right=self.normalize(root.right, sample_size, total_size)
            
            return left + right
            
        elif isinstance(root, LeafNode):
            
            size = root.getSize()
            new_size = (size/ total_size) * sample_size
            
            root.setSize( new_size )
            
            return size
        
        return 0
    
    def count_nodes(self, root):
        total_count = 0
        decision_count=0
        leaf_count=0
        data_count = 0
        
        stack = [root]
        while stack:
            node = stack.pop()
            total_count += 1
            if isinstance(node, DecisionNode):
                stack.append(node.right)
                stack.append(node.left)
                
                decision_count+=1
            
            elif isinstance(node, LeafNode):
                data_count+=node.getSize()
                leaf_count+=1
                
        return total_count, decision_count, leaf_count, data_count
        
    def PRINT(self, root, prefix=""):
        if isinstance(root, LeafNode):
            print(prefix + "+- <ExNode>")
            print(prefix + "=> ",root.getSize())
            return;
        
        elif isinstance(root, DecisionNode):
            left = root.left
            right = root.right
            print(prefix +"+- <InNode>")
            print(prefix +"=> ",root.getAtt())
            self.PRINT(right,prefix+"|R \t")
            self.PRINT(left,prefix+"|L \t")
    
    def clear(self, root):
        if isinstance(root, DecisionNode):
            
            left = root.left
            right = root.right
            
            self.clear(left)
            self.clear(right)
        
        elif isinstance(root, LeafNode):
            root.setSize(0)
            
        
class IsolationForest_cum:
    
    def __init__(self, sample_size, n_trees, n_clients):
        
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.n_clients=n_clients

    def getTrees(self):
        return self.trees
    
    def print_splitCount(self):
        global split_count
        count = split_count
        split_count=0
        
        return count
        
    def count_nodes(self):
        for i, tree in enumerate(self.trees):
            total_count, decision_count, leaf_count, data_count = tree.count_nodes(tree.root)
            print(f"tree[{i}] {total_count, decision_count, leaf_count, data_count}")
            
    def normalize(self):
        for tree in self.trees:
            
            new_total = tree.normalize(tree.root, self.sample_size, tree.total_leafsize)
            
            tree.total_leafsize = math.ceil(new_total)

    def supply(self, X):
        for tree in self.trees:
            new_total=0
            for x in X:
                x_sub=sub_sample(x, self.sample_size)
                new_total=tree.supply(tree.root, x_sub)
                
            tree.total_leafsize = new_total
            
    def PRINT(self):
        for i, tree in enumerate(self.trees):
            print(f"= = = = {i} TREE = = = =")
            tree.PRINT(tree.root)
    
    def clear(self):
        for tree in self.trees:
            tree.clear(tree.root)
            tree.total_leafsize=0

    def grow(self, X, new_height):

        np.random.shuffle(X)
        for i in range(self.n_trees):
            j = math.floor(i * 1.0 / self.per)
            tree = self.trees[i]

            x = X[j]
            n_rows = x.shape[0]

            data_index = np.random.permutation(n_rows)
            data_index = data_index[:self.sample_size]

            X_sub = x[data_index]

            tree.setheightlimit(new_height)

            tree.grow(X_sub)

        self.height_limit = new_height


    def fit(self, X: np.ndarray, init_height):

        self.trees = []
        if isinstance(X, pd.DataFrame):
            X = X.values

        self.height_limit = init_height
        self.per = math.ceil(self.n_trees / self.n_clients)
        for i in range(self.n_trees):
            j = math.floor(i * 1.0 / self.per)
            x = X[j]

            n_rows = x.shape[0]

            data_index = np.random.permutation(n_rows)
            data_index = data_index[:self.sample_size]

            X_sub = x[data_index]

            tree = IsolationTree(0, self.height_limit)

            tree.fit(X_sub, 0)

            self.trees.append(tree)

        return self

    def path_length(self, X: np.ndarray) -> np.ndarray:

        paths = []
        for row in X:
            path = []
            for tree in self.trees:
                node = tree.root
                length = 0
                while isinstance(node, DecisionNode):
                    if row[node.splitAtt] < node.splitVal:
                        node = node.left
                    else:
                        node = node.right

                    length += 1
                leaf_size = node.size
                pathLength = length + c(leaf_size)
                path.append(pathLength)

            paths.append(path)

        paths = np.array(paths)
        out = np.mean(paths,axis=1)
            
        return out
        
    def anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values

        avg_length = self.path_length(X)

        scores = np.array([np.power(2, -l/c(self.sample_size))
                            for l in avg_length])
        
        return scores

    def predict_from_anomaly_scores(self, scores: np.ndarray, threshold:float) -> np.ndarray:
        return np.array([1 if s >= threshold else 0 for s in scores])

    def predict(self, X: np.ndarray, doAvg:bool, threshold:float) -> np.ndarray:
        
        scores = self.anomaly_score(X, doAvg)
        
        prediction = self.predict_from_anomaly_scores(scores, threshold)
        
        return prediction
    
def find_TPR_threshold(label, scores, desired_TPR):

    TPR = 0
    FPR = 0
    threshold = 1
    while TPR < desired_TPR:
        threshold -= 0.01
        prediction = [1 if s > threshold else 0 for s in scores]
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for p, y in zip(prediction, label):
            if p == 1 and y == 1:
                TP += 1
            if p == 0 and y == 0:
                TN += 1
            if p == 1 and y == 0:
                FP += 1
            if p == 0 and y == 1:
                FN += 1
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        
        if threshold < 0:
            print("The model cannot reach the desired TPR")
            return

    return threshold, FPR        

def sub_sample(X, sample_size):
    data_index = np.random.permutation(X.shape[0])
    
    data_index = data_index[:sample_size]
    
    X_sub=X[data_index]
    
    return X_sub
