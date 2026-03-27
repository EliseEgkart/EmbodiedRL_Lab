# Chapter 8: Deep Q-Network

## 왜 순진한 신경망 Q-learning은 불안정한가

신경망과 Q-learning을 그대로 합치면 다음 문제가 동시에 생깁니다.

- 연속된 샘플들이 강하게 상관됨
- 타깃도 같은 네트워크에서 만들어짐
- 부트스트랩 오차가 증폭될 수 있음

DQN은 이를 replay buffer와 target network로 완화합니다.

## DQN 타깃

전이 `(s, a, r, s', done)`에 대해 타깃은

`y = r + (1 - done) * gamma * max_{a'} Q_{theta^-}(s', a')`

입니다.

- `theta`: 온라인 네트워크
- `theta^-`: 타깃 네트워크

손실은

`L(theta) = E[(Q_theta(s, a) - y)^2]`

입니다.

[dqn.py](./dqn.py)와 [../pytorch/dqn.py](../pytorch/dqn.py)가 같은 구조를 서로 다른 프레임워크로 구현합니다.

## 경험 재현(Replay Buffer)

Replay buffer는 전이를 저장해 두고 무작위 미니배치를 뽑습니다.

수학적 이유는 다음과 같습니다.

- SGD는 i.i.d.에 가까운 샘플을 좋아함
- 연속 전이 상관을 약화시킴
- 한 경험을 여러 번 재사용 가능

[replay_buffer.py](./replay_buffer.py)가 이 메커니즘만 따로 보여줍니다.

## 타깃 네트워크

예측 네트워크와 타깃 네트워크가 같으면, 파라미터를 한 번 갱신할 때마다 레이블도 같이 움직이는 비정상 회귀문제가 됩니다.

그래서 DQN은

- 온라인 네트워크는 계속 학습
- 타깃 네트워크는 주기적으로만 동기화

하는 구조를 씁니다.

## Bellman residual 최소화

DQN은 샘플된 전이에 대해 Bellman residual을 근사적으로 줄이는 과정으로 볼 수 있습니다. 다만 지도학습과 달리 정답 레이블이 외부에서 주어지지 않고, 현재 에이전트가 만든 bootstrap target이라는 점이 본질적으로 어렵습니다.

## CartPole의 의미

CartPole은

- 상태가 연속 벡터
- 행동은 이산 2개
- 살아 있는 동안 보상이 누적

인 환경입니다. 따라서 표 기반 Q-learning은 불가능하고 함수근사가 필수입니다.

## epsilon-greedy는 여전히 중요하다

심층강화학습이라고 해서 탐험 원리가 갑자기 복잡해지는 것은 아닙니다. 이 코드에서는 여전히 epsilon-greedy를 사용합니다.

- `epsilon` 확률로 랜덤
- 아니면 네트워크 argmax

작은 이산 행동환경에서는 이것만으로도 충분한 경우가 많습니다.

## detach가 왜 필요한가

타깃 경로에서 gradient를 끊지 않으면 모델이 예측값과 타깃을 동시에 움직이려 하게 됩니다. 그러면 Bellman 회귀 해석이 깨집니다. 그래서 `next_q`는 반드시 detach되어야 합니다.

## 구현 통찰

- `np.stack`은 리스트 전이를 미니배치 텐서로 바꾸는 핵심 단계입니다.
- sync interval은 타깃의 최신성과 안정성 사이 절충입니다.
- replay buffer 크기는 학습 분포 다양성에 영향을 줍니다.
- DQN은 탐험 데이터를 쓰지만 greedy 정책을 학습하므로 본질적으로 off-policy입니다.

## 한계

Vanilla DQN은 여전히

- max 연산으로 인한 과대추정
- 하이퍼파라미터 민감성
- 어려운 환경에서의 불안정성

문제가 있습니다. 이후 확장 기법들이 이를 보완합니다.
