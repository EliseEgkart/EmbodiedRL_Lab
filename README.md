# EmbodiedRL Lab Study Index

이 저장소의 `deep-learning-from-scratch-4` 학습 자료를 장별로 정리한 상위 인덱스입니다. 각 챕터는 영문 정리 `README.md`와 한글 정리 `README_ko.md`를 함께 제공합니다.

## 구성

- 교재 흐름 기준: `ch01` ~ `ch09`
- 보조 구현: `common`, `pytorch`
- 이론 보강: `ch02`, `ch03`은 실행 코드 대신 수학 정리 문서 중심

## 빠른 링크

| Chapter | 주제 | English | 한국어 |
|---|---|---|---|
| ch01 | Bandit / 탐험-활용 | [README](./deep-learning-from-scratch-4/ch01/README.md) | [README_ko](./deep-learning-from-scratch-4/ch01/README_ko.md) |
| ch02 | Markov Process / MDP 기초 | [README](./deep-learning-from-scratch-4/ch02/README.md) | [README_ko](./deep-learning-from-scratch-4/ch02/README_ko.md) |
| ch03 | Bellman Equation / 최적성 | [README](./deep-learning-from-scratch-4/ch03/README.md) | [README_ko](./deep-learning-from-scratch-4/ch03/README_ko.md) |
| ch04 | Dynamic Programming / 동적계획법 | [README](./deep-learning-from-scratch-4/ch04/README.md) | [README_ko](./deep-learning-from-scratch-4/ch04/README_ko.md) |
| ch05 | Monte Carlo / 반환값 기반 학습 | [README](./deep-learning-from-scratch-4/ch05/README.md) | [README_ko](./deep-learning-from-scratch-4/ch05/README_ko.md) |
| ch06 | Temporal Difference / TD, SARSA, Q-learning | [README](./deep-learning-from-scratch-4/ch06/README.md) | [README_ko](./deep-learning-from-scratch-4/ch06/README_ko.md) |
| ch07 | Neural Network Approximation / 함수근사 | [README](./deep-learning-from-scratch-4/ch07/README.md) | [README_ko](./deep-learning-from-scratch-4/ch07/README_ko.md) |
| ch08 | DQN / Deep Q-Learning | [README](./deep-learning-from-scratch-4/ch08/README.md) | [README_ko](./deep-learning-from-scratch-4/ch08/README_ko.md) |
| ch09 | Policy Gradient / Actor-Critic | [README](./deep-learning-from-scratch-4/ch09/README.md) | [README_ko](./deep-learning-from-scratch-4/ch09/README_ko.md) |

## 챕터별 안내

### ch01

- 밴딧 문제, 표본평균, epsilon-greedy, 비정상 환경에서의 상수 보폭을 다룹니다.
- 시작점 파일: [bandit.py](./deep-learning-from-scratch-4/ch01/bandit.py), [non_stationary.py](./deep-learning-from-scratch-4/ch01/non_stationary.py)

### ch02

- 코드보다 이론 연결이 목적입니다.
- 밴딧에서 MDP로 넘어가기 위한 마코프 성질, 상태가치, 행동가치 정의를 정리합니다.

### ch03

- Bellman 기대 방정식, Bellman 최적 방정식, 정책개선 정리를 정리합니다.
- Chapter 4 이후 코드를 읽기 위한 수학 기반 역할을 합니다.

### ch04

- GridWorld에서 정책평가, 정책반복, 가치반복을 구현합니다.
- 시작점 파일: [policy_eval.py](./deep-learning-from-scratch-4/ch04/policy_eval.py), [policy_iter.py](./deep-learning-from-scratch-4/ch04/policy_iter.py), [value_iter.py](./deep-learning-from-scratch-4/ch04/value_iter.py)

### ch05

- Monte Carlo 평가와 제어, off-policy Monte Carlo, importance sampling을 다룹니다.
- 시작점 파일: [mc_eval.py](./deep-learning-from-scratch-4/ch05/mc_eval.py), [mc_control.py](./deep-learning-from-scratch-4/ch05/mc_control.py), [mc_control_offpolicy.py](./deep-learning-from-scratch-4/ch05/mc_control_offpolicy.py)

### ch06

- TD(0), SARSA, off-policy SARSA, Q-learning을 다룹니다.
- 시작점 파일: [td_eval.py](./deep-learning-from-scratch-4/ch06/td_eval.py), [sarsa.py](./deep-learning-from-scratch-4/ch06/sarsa.py), [q_learning.py](./deep-learning-from-scratch-4/ch06/q_learning.py)

### ch07

- DeZero 기반 자동미분, 선형회귀, 다층퍼셉트론, 신경망 Q-learning을 다룹니다.
- 시작점 파일: [dezero3.py](./deep-learning-from-scratch-4/ch07/dezero3.py), [dezero4.py](./deep-learning-from-scratch-4/ch07/dezero4.py), [q_learning_nn.py](./deep-learning-from-scratch-4/ch07/q_learning_nn.py)

### ch08

- CartPole에 대한 DQN, replay buffer, target network를 다룹니다.
- 시작점 파일: [dqn.py](./deep-learning-from-scratch-4/ch08/dqn.py), [replay_buffer.py](./deep-learning-from-scratch-4/ch08/replay_buffer.py)

### ch09

- simple policy gradient, REINFORCE, actor-critic을 다룹니다.
- 시작점 파일: [simple_pg.py](./deep-learning-from-scratch-4/ch09/simple_pg.py), [reinforce.py](./deep-learning-from-scratch-4/ch09/reinforce.py), [actor_critic.py](./deep-learning-from-scratch-4/ch09/actor_critic.py)

## 공통 구현 자료

- 환경 및 시각화: [common/gridworld.py](./deep-learning-from-scratch-4/common/gridworld.py), [common/gridworld_render.py](./deep-learning-from-scratch-4/common/gridworld_render.py)
- 공통 유틸: [common/utils.py](./deep-learning-from-scratch-4/common/utils.py)
- PyTorch 대응 구현: [pytorch/simple_pg.py](./deep-learning-from-scratch-4/pytorch/simple_pg.py), [pytorch/reinforce.py](./deep-learning-from-scratch-4/pytorch/reinforce.py), [pytorch/actor_critic.py](./deep-learning-from-scratch-4/pytorch/actor_critic.py), [pytorch/dqn.py](./deep-learning-from-scratch-4/pytorch/dqn.py)

## 추천 읽기 순서

1. `ch01`로 탐험-활용과 증분업데이트 감각을 잡습니다.
2. `ch02`, `ch03`로 MDP와 Bellman 수학을 먼저 정리합니다.
3. `ch04`에서 모델 기반 정확한 백업을 봅니다.
4. `ch05`, `ch06`에서 샘플 기반 MC와 TD를 비교합니다.
5. `ch07`, `ch08`에서 함수근사와 DQN으로 확장합니다.
6. `ch09`에서 정책 직접 최적화 관점으로 넘어갑니다.
