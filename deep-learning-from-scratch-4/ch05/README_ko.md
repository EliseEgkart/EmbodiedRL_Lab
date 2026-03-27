# Chapter 5: Monte Carlo 방법

## 핵심 아이디어

Monte Carlo는 환경 모델의 정확한 기대값을 쓰지 않고, 실제로 생성된 전체 에피소드의 반환값을 직접 평균내서 가치를 추정합니다.

## 반환값

시각 `t`의 반환값은

`G_t = R_{t+1} + gamma R_{t+2} + gamma^2 R_{t+3} + ...`

입니다.

Monte Carlo는 이 `G_t`를 직접 타깃으로 사용합니다.

## 정책 평가

고정 정책 `pi`에 대해

`V^pi(s) = E_pi[G_t | S_t = s]`

이므로, 실제 에피소드에서 상태 방문 후 관측된 반환값들을 평균내면 `V^pi`를 추정할 수 있습니다.

[mc_eval.py](./mc_eval.py)는 에피소드를 `memory`에 저장해 두었다가 끝난 뒤 뒤에서부터 `G`를 누적합니다.

## 왜 에피소드 전체를 저장하는가

Monte Carlo는 한 전이만으로는 업데이트할 수 없습니다. 시각 `t`의 타깃 `G_t`는 종료까지의 미래 보상을 모두 알아야 하기 때문입니다.

그래서 구조가 항상 다음과 같습니다.

1. 한 에피소드 수집
2. 뒤에서부터 반환값 계산
3. 평균 혹은 보폭 업데이트

## Monte Carlo 제어

행동가치로 바꾸면

`Q(s, a) = E[G_t | S_t = s, A_t = a]`

를 추정하게 됩니다.

[mc_control.py](./mc_control.py)는

`Q <- Q + alpha * (G - Q)`

형태의 상수 보폭 업데이트를 사용합니다. 엄밀한 표본평균보다 온라인 구현이 쉽고, 비정상성에도 더 잘 적응할 수 있습니다.

## Off-policy와 중요도 샘플링

데이터는 행동정책 `b`로 모았지만, 목표는 다른 정책 `pi`를 평가하거나 개선하는 경우가 있습니다.

이때 중요도 비율

`rho_{t:T-1} = product_{k=t}^{T-1} pi(A_k | S_k) / b(A_k | S_k)`

를 사용하면

`E_b[rho G_t] = E_pi[G_t]`

가 됩니다.

[importance_sampling.py](./importance_sampling.py)는 작은 이산 예제로 이를 먼저 보여주고, [mc_control_offpolicy.py](./mc_control_offpolicy.py)에서 제어 문제로 확장합니다.

## 편향-분산 관점

Monte Carlo 타깃은 원칙적으로 편향이 작거나 없지만 분산이 큽니다.

- 에피소드가 길수록 노이즈가 누적
- 보상과 전이의 확률성 반영
- off-policy에서는 중요도 비율이 폭주 가능

그래서 다음 장에서 TD가 등장합니다.

## 구현 통찰

- 뒤에서 앞으로 가면 `G_t = R_{t+1} + gamma G_{t+1}`를 그대로 코드화할 수 있습니다.
- epsilon-greedy 정책개선은 이미 generalized policy iteration의 형태입니다.
- constant-alpha 업데이트는 순수 평균보다 실전 코드에서 더 유연합니다.
- off-policy MC는 수학적으로 깔끔하지만 수치적으로는 매우 불안정할 수 있습니다.

## 다음 장 연결

TD는 Monte Carlo의 전체 반환값 `G_t` 대신

`R_{t+1} + gamma V(S_{t+1})`

같은 1-step 부트스트랩 타깃을 사용하여 더 빠르고 저분산의 업데이트를 수행합니다.
