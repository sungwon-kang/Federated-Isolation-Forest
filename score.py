import argparse
import numpy as np
from utils import util
from sklearn.metrics import roc_auc_score
from iForest_cum import IsolationForest_cum
from iForest import IsolationForest

def score(train, test, label, args):

    height_limit = args.height_limit
    sampling_size = args.sampling_size
    n_clients = args.n_clients
    n_trees = args.n_trees

    if args.method == 'clientwise': # Fed Clientwise iForest

        iF = IsolationForest(sampling_size, n_trees, n_clients)
        iF.fit(train, height_limit)

    else:
        if args.method == 'levelwise': # Fed levelwise iForest
            iF = IsolationForest(sampling_size, n_trees, n_clients)
        if args.method == 'levelwise_cum': # Fed levelwise iForest cum
            iF = IsolationForest_cum(sampling_size, n_trees, n_clients)

        iF.fit(train, 1)
        for new_height in range(2, height_limit + 1):
            iF.grow(train, new_height)

    scores=iF.anomaly_score(test)
    roc_auc = roc_auc_score(label, scores)
    print("ROC AUC Score: {}".format(roc_auc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated_XGBoost_WbyT')
    parser.add_argument('--seed', default=0, type=int, help='랜덤 시드 설정')
    parser.add_argument('--filename', default="pendigits", type=str, help='데이터 파일명')
    parser.add_argument('--test_size', default=0.2, type=float, help='데이터 파일명')
    parser.add_argument('--n_clients', default=10, type=int, help='클라이언트 수')
    parser.add_argument('--isiid', default=True, type=bool, help='클라이언트 데이터 환경 설정')
    parser.add_argument('--alpha', default=2, type=int, help='Non-IID 데이터 환경에서 불균형 조절 매개변수')
    parser.add_argument('--outlier', default=9, type=int, help='데이터 셋의 이상치 클래스 설정')

    parser.add_argument('--n_trees', default=100, type=int, help='의사 결정 트리의 수')
    parser.add_argument('--height_limit', default=10, type=int, help='트리의 최대 깊이')
    parser.add_argument('--sampling_size', default=256, type=int, help='트리 당 샘플링 사이즈 비율')
    parser.add_argument('--method', default='levelwise', type=str, help="트리 학습 방법", choices=['clientwise', 'levelwise', 'levelwise_cum'])
    args = parser.parse_args()

    setter = util(seed=args.seed, alpha=args.alpha, isiid=args.isiid, n_clients=args.n_clients)
    train, test, label = setter.getdata(args.filename, args.outlier, args.test_size)
    score(train, test, label, args)