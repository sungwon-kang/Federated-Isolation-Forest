import numpy as np
import pandas as pd
import math


# iTree의 n개의 노드를 가질 때, 평균 path length 추정치
def c(size):
    if size > 2:
        return 2 * (np.log(size-1)+0.5772156649) - 2*(size-1)/size
    if size == 2:
        return 1
    return 0

# external 노드
class LeafNode:
    def __init__(self, size, data):
        self.size = size
        self.data = data
        
    def SET(self, size, data):
        self.size=size
        self.data=data
        
    def GETsize(self):
        return self.size
    
    def GETdata(self):
        return self.data
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
        
    def setheightlimit(self, h):
        self.height_limit = h
    
    def getRoot(self):
        return self.root
    
    def fit(self, X: np.ndarray):

        # 논문에 있는 Original 코드
        if self.height >= self.height_limit or X.shape[0] <= 2:
            self.root = LeafNode(X.shape[0], X)
            return self.root

        num_features = X.shape[1]  # 속성의 갯수(열)을 저장
        splitAtt = np.random.randint(0, num_features)  # 0 ~ 속성의 갯수를 랜덤으로 정함
        splitVal = np.random.uniform(min(X[:, splitAtt]), max(X[:, splitAtt]))  # 선택된 속성의 min max 사이의 값 저장

        X_left = X[X[:, splitAtt] < splitVal]
        X_right = X[X[:, splitAtt] >= splitVal]

        left = IsolationTree(self.height + 1, self.height_limit)
        right = IsolationTree(self.height + 1, self.height_limit)

        # 재귀 호출하여 진행
        left.fit(X_left)
        right.fit(X_right)

        # 현재 internal 노드의 정보를 저장
        self.root = DecisionNode(left.root, right.root, splitAtt, splitVal)

        return self.root


    def grow(self, X):
        if self.height >= self.height_limit or X.shape[0] <= 2: 
            if isinstance(self.root, DecisionNode) or isinstance(self.root, LeafNode):
                return self.root

            
        left = IsolationTree(self.height + 1, self.height_limit)
        right = IsolationTree(self.height + 1, self.height_limit) 
        
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
            
            # 완전히 같을려면 h층의 모델의 h-1층과 같아야한다.
            X_left = X[X[:, splitAtt] < splitVal]
            X_right = X[X[:, splitAtt] >= splitVal]  

            # 재귀 호출하여 진행
            left.fit(X_left)
            right.fit(X_right)

            # 현재 internal 노드의 정보를 저장
            self.root = DecisionNode(left.root, right.root, splitAtt, splitVal)
        
        return self.root

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
                data_count+=node.GETsize()
                leaf_count+=1
        return total_count, decision_count, leaf_count, data_count

class IsolationForest:
    
    def __init__(self, sample_size, n_trees, K):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.K=K
    
    def getTrees(self):
        return self.trees
        
        
    def PRINT(self, root, prefix):
        if isinstance(root, LeafNode):
            print(prefix + "+- <ExNode>")
            print(prefix + "=> ",root.GETsize())
            return;
        
        print(prefix +"+- <InNode>")
        print(prefix +"=> ",root.getAtt())
        self.PRINT(root.right, prefix+"|R \t")
        self.PRINT(root.left, prefix+"|L \t")
        
    def grow(self, X, new_height):

        np.random.shuffle(X)
        for i in range(self.n_trees):
            j = math.floor(i*1.0/self.per)
            tree = self.trees[i]

            x = X[j]
            n_rows=x.shape[0]
            
            data_index = np.random.permutation(n_rows)
            data_index = data_index[:self.sample_size]
                
            X_sub=x[data_index]
            
            tree.setheightlimit(new_height)
            
            tree.grow(X_sub)
            
        self.height_limit=new_height

    def fit(self, X: np.ndarray, init_height):

        self.trees = []
        if isinstance(X, pd.DataFrame):
            X = X.values  # X의 모든 값들을 X에 저장.

        self.height_limit = init_height
        self.per = math.ceil(self.n_trees/self.K)
        for i in range(self.n_trees):
            j = math.floor(i*1.0/self.per)
            x = X[j]
            
            n_rows = x.shape[0]  # X의 데이터 수 저장

            data_index = np.random.permutation(n_rows)
            data_index = data_index[:self.sample_size]
            
            # data_index에 해당되는 데이터(행)들을 X_sub에 저장.
            X_sub = x[data_index]
            
            # 현재 높이 0, 높이 한계를 넘겨주며 iTree 객체 생성.
            tree = IsolationTree(0, self.height_limit) # 루트 노드만 생성 
            
            # iTree 학습
            tree.fit(X_sub)

            # 완성된 iTree append
            self.trees.append(tree)
        
        return self

    # path length 계산
    def path_length(self, X: np.ndarray) -> np.ndarray:

        # 모든 관측치에 대한 모든 iTree의 path length를 저장할 리스트
        paths = []
        # inputdata X에 있는 데이터들만큼 반복
        for row in X:
            # 한 데이터에 대한 iTree의 모든 path length를 저장할 리스트
            path = []  # paths= [ path1, path2, ... ]
            # 완성된 iTree마다 반복
            for tree in self.trees:
                # 트리의 최상단 root
                node = tree.root
                length = 0
                # internal 노드 반복
                while isinstance(node, DecisionNode):
                    # 해당 노드의 속성의 분할 값보다 크다면
                    if row[node.splitAtt] < node.splitVal:
                        # 노드를 왼쪽으로 순회
                        node = node.left
                    else:
                        # 아니면 오른쪽으로 순회
                        node = node.right

                    # 순회하면서 path length를 1 증가
                    length += 1
                # 순회가 종료되고 external node를 만났다면,
                # node에 고립된 데이터의 크기를 leaf_size에 저장
                leaf_size = node.size

                # path lnegth 계산
                pathLength = length + c(leaf_size)

                # 한 관측치 X에 대한 한 iTree의 pathlength 저장.
                # tree의 갯수만큼 다시 반복
                path.append(pathLength)

            # 모든 iTree에 대한 path length가 저장된 path를 paths에 삽입.
            paths.append(path)

        # numpy로 변환
        paths = np.array(paths)

        # X에 있는 각 관측치에 대한 평균 path length를 (n,1)형태로 반환
        return np.mean(paths, axis=1)
    
    # def path_length(self, tree: IsolationTree, X: np.ndarray) -> np.ndarray:
        
    #     for tree in self
    #         self.path_length(tree, X)
        
    def anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values

        avg_length = self.path_length(X)
        scores = np.array([np.power(2, -l/c(self.sample_size))
                          for l in avg_length])
        return scores

    def predict_from_anomaly_scores(self, scores: np.ndarray, threshold:float) -> np.ndarray:
        # Anomaly score가 threshold 이상인 s에 대한 관측치에만 Anomaly로 취급
        return np.array([1 if s >= threshold else 0 for s in scores])

    def predict(self, X: np.ndarray, threshold:float) -> np.ndarray:
        
        # 인스턴스 X에 대하여 anomaly score를 계산한다.
        scores = self.anomaly_score(X)
        # 계산된 Anomaly score를 threshold로 Anomaly를 구별한다.
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
