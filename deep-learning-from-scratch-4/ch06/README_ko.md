# Chapter 6: Temporal-Difference 학습

## 핵심 아이디어

TD는 에피소드가 끝날 때까지 기다리지 않고, 한 단계 전이만으로 즉시 업데이트합니다. 즉, Monte Carlo의 “완전한 반환값”을 “부트스트랩된 근사 타깃”으로 바꿉니다.

## TD(0) 예측

정책 평가에서 TD(0) 타깃은

`R_{t+1} + gamma V(S_{t+1})`

이고, 업데이트는

`V(S_t) <- V(S_t) + alpha [R_{t+1} + gamma V(S_{t+1}) - V(S_t)]`

입니다.

[td_eval.py](./td_eval.py)가 이 식을 그대로 구현합니다.

대괄호 안 오차는 TD error

`delta_t = R_{t+1} + gamma V(S_{t+1}) - V(S_t)`

입니다.

## Monte Carlo와의 차이

- Monte Carlo: 편향 적음, 분산 큼, 에피소드 종료 필요
- TD: 편향 있음, 분산 낮음, 즉시 업데이트 가능

즉, 전형적인 bias-variance tradeoff입니다.

## SARSA

SARSA의 타깃은

`R_{t+1} + gamma Q(S_{t+1}, A_{t+1})`

입니다. 이름도 `(S, A, R, S, A)`에서 나왔습니다.

[sarsa.py](./sarsa.py)는 실제로 다음 행동 `A_{t+1}`이 샘플링되어야 이전 상태를 업데이트할 수 있으므로 길이 2짜리 버퍼를 사용합니다.

SARSA는 on-policy이므로 탐험의 위험까지 포함한 정책 가치를 반영하는 경향이 있습니다.

## Q-learning

Q-learning의 타깃은

`R_{t+1} + gamma max_a Q(S_{t+1}, a)`

입니다.

즉, 실제로 다음에 어떤 행동을 했는지가 아니라 “최적으로 행동한다면”의 값을 학습합니다. 그래서 off-policy입니다.

[q_learning.py](./q_learning.py)는

- 목표 정책 `pi`는 greedy
- 행동 정책 `b`는 epsilon-greedy

로 분리해서 관리합니다.

## Off-policy SARSA

[sarsa_off_policy.py](./sarsa_off_policy.py)는 중요도 비율

`rho = pi(A_{t+1} | S_{t+1}) / b(A_{t+1} | S_{t+1})`

를 사용해 1-step 타깃을 보정합니다. 이는 Chapter 5의 중요도 샘플링이 TD 형태로 들어온 것이라고 볼 수 있습니다.

## 부트스트래핑

TD의 핵심은 현재 추정값을 다른 추정값으로 업데이트한다는 점입니다.

- `V(s)`를 `V(s')`로 보정
- `Q(s, a)`를 `Q(s', a')` 혹은 `max_a Q(s', a)`로 보정

이 덕분에 빠르게 학습하지만, 동시에 오차가 전파될 수 있습니다.

## 구현 통찰

- 종료 상태에서는 bootstrap 항이 `0`입니다.
- SARSA의 마지막 더미 업데이트는 마지막 실제 전이를 종료 타깃으로 닫기 위한 구현 장치입니다.
- `greedy_probs`는 정책개선을 매 스텝 암묵적으로 수행합니다.
- Q-learning이 SARSA보다 더 공격적으로 보이는 이유는 실제 탐험행동이 아니라 최댓값을 타깃으로 쓰기 때문입니다.

## 다음 장 연결

여기까지는 표 형태로 `V`, `Q`를 저장할 수 있었습니다. 하지만 상태가 커지거나 연속화되면 표가 불가능해지므로, Chapter 7에서는 신경망 함수근사로 넘어갑니다.
