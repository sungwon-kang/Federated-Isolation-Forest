# 연합 Isolation Forest

이 레포지토리는 Non-IID 데이터 분포에서 성능 저하를 극복하기 위한 연합 Isolation Forest[1]를 구현한 코드를 저장합니다.
기본 알고리즘은 [1]의 새로운 연합학습 방법을 제안하였습니다.

[1] S. Kang, C. Park, "Federated Isolation Forest," Journal of Korea Multimedia Society Vol. 27, No. 1, pp. 159-169, 2024.

(paper: https://doi.org/10.9717/kmms.2024.27.1.159)

[2] F.T. Liu, K.M. Ting, and Z.H. Zhou, “Isolation Forest,” P roceedings of 2008 Eighth IEEE International Conference on Data Mining, pp.
413-422, 2008

(paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4781136)


## 소개

구현된 연합 Isolation Forest(이하 연합 iForest)는 Non-IID 데이터를 갖는 클라이언트들의 데이터에서 이상치를 탐색할 수 있습니다.
이 연합 iForest의 Isolation tree(iTree)는 각 클라이언트들을 순회하면서 트리의 층을 성장시켜 완성합니다. 
이러한 방법을 통해 모든 클라이언트들의 데이터 특성을 담은 트리를 생성할 수 있습니다. 이 방법을 적용한 모델을 [1]에서는 Fed levelwise iForest라고 칭하였습니다.

iForest의 이상치 탐지는 리프노드 크기와 트리들의 평균 높이를 바탕으로 계산되는 이상치 스코어에 의해 수행됩니다. Fed levelwise iForest cum 모델은 iTree이 성장하면서 리프노드 크기를 저장하는 방식을 개선합니다. 
Fed levelwise iForest는 트리의 층이 성장하면서 생성된 리프노드는 현재 클라이언트에 의해 결정노드로 대체되어 리프노드 크기가 소멸됩니다. 이 문제점을 해결하기 위해 제안된 모델이 Fed levelwise iForest cum 입니다.
모델의 자세한 알고리즘은 [1] 논문을 참조해주시길 바랍니다.

이 프로젝트에서 구현된 모델은 다음과 같습니다.
1. Fed clientwise iForest
2. Fed levelwise iForest
3. Fed levelwise iForest cum

## 설치

1. `python3.9.12`와 필요한 라이브러리 및 패키지 설치하세요.
   ```
   pip install -r requirements.txt
   ```

## 디렉토리 및 파일 구조

- `score.py`: 연합학습 실험을 실행하기 위한 파일 
- `iForest.py`: Fed levelwise iForest 구현 코드
- `iForest_cum.py`: Fed levelwise iForest cum 구현 코드
- `utils`: 연합학습 환경을 조성하기 위한 여러 유틸기능 제공.
- `data/`: 데이터 셋이 저장된 디렉토리

## 실험환경 설정변수
- `--seed`: 재현성을 위한 랜덤 시드를 설정.
  - Default: 0
- `--filename`: 훈련에 사용할 데이터 파일(csv)의 이름.
  - Default: pendigits
- `--test_size`: (기본값: 0.2, 타입: float): 테스트에 사용할 데이터의 비율.
  - Default: 0.2
- `--n_clinets`: 연합학습 참여 클라이언트 수
  - Default: 10
  
- `--isiid`: 클라이언트 간 데이터 분포가 IID(독립적이고 동일한 분포)인지 여부를 지정.
  - Default: True
   
- `--alpha`: Non-IID 데이터 환경에서 불균형을 조절하는 매개변수로, 각 클래스마다 alpha개의 청크로 나누어 클라이언트에게 랜덤하게 할당.
  - Default: 2

- `--outlier`: 데이터셋에서 이상치 클래스를 지정.
  - Default: 9

## Isolation Forest 학습 설정변수
- `--n_trees`: 앙상블 모델의 의사결정 나무 수.
  - Default: 100

- `--height_limit`: 각 의사결정 나무의 최대 깊이입니다.
  - Default: 10

- `--sampling_size`: 각 나무를 훈련하는 데 사용되는 샘플 수를 나타내는 샘플링 크기.
  - Default: 256

- `--method`: 나무를 훈련하는 데 사용되는 방법. 가능한 선택지는 다음과 같습니다:
  - Default: levelwise_cum
  1. clientwise: 각 클라이언트가 독립적으로 나무를 성장시킵니다.
  2. levelwise: 나무를 레벨 단위로 성장시키며 클라이언트 간의 협력이 필요합니다.
  3. levelwise_cum: levelwise와 유사하지만 층이 성장할 때마다 리프노드 크기를 누적합니다.

## 사용법
다음 명령어를 통해 코드를 실행:
```
python score.py
```