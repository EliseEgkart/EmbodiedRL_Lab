# Chapter 9: 정책경사와 Actor-Critic

## 왜 가치기반을 넘어서려 하는가

가치기반 방법은 `Q`를 학습한 뒤 `argmax`로 행동을 고릅니다. 반면 정책경사는 정책 자체를 직접 최적화합니다. 다음 상황에서 특히 유리합니다.

- 확률적 정책이 자연스러울 때
- 정책을 직접 매개변수화하는 편이 쉬울 때
- 행동분포 자체를 학습하고 싶을 때

## 목적함수

정책이 `pi_theta(a | s)`일 때 에피소드 목적함수는

`J(theta) = E_{tau ~ pi_theta}[G_0]`

입니다.

문제는 trajectory 자체가 `theta`에 의존한다는 점입니다.

## 로그미분 트릭

다음 항등식

`grad pi_theta = pi_theta grad log pi_theta`

를 쓰면

`grad_theta J(theta) = E[sum_t grad_theta log pi_theta(A_t | S_t) G_t]`

를 얻습니다. 이것이 REINFORCE의 기반입니다.

## 단순 정책경사

[simple_pg.py](./simple_pg.py)는 에피소드 전체 반환값 `G_0`를 모든 시점에 공통으로 사용합니다.

`sum_t grad log pi_theta(A_t | S_t) G_0`

이는 유효하지만 분산이 큽니다.

## REINFORCE

[reinforce.py](./reinforce.py)는 reward-to-go를 사용합니다.

`sum_t grad log pi_theta(A_t | S_t) G_t`

즉, 행동 이후의 미래보상만 사용하므로 그 행동이 영향을 줄 수 없는 과거 보상을 제거합니다. 편향은 유지하지 않으면서 분산을 줄이는 방향입니다.

## baseline과 advantage

행동과 무관한 baseline `b(s)`를 빼도 정책경사는 편향되지 않습니다.

`E[grad log pi(a | s) * b(s)] = 0`

대표적인 baseline은 `V(s)`이고, 이때 advantage는

`A(s, a) = Q(s, a) - V(s)`

입니다.

1-step actor-critic에서는 흔히

`delta_t = R_{t+1} + gamma V(S_{t+1}) - V(S_t)`

를 advantage 근사로 씁니다.

## Actor-Critic

[actor_critic.py](./actor_critic.py)는

- actor: 정책망
- critic: 가치망

을 함께 학습합니다.

critic 손실은

`L_V = (V(s) - [r + gamma V(s')])^2`

이고, actor 손실은

`L_pi = -log pi(a | s) * delta`

형태입니다.

즉, critic이 낮은 분산의 학습 신호를 actor에 제공합니다.

## 편향-분산 구조

- simple PG: 편향 거의 없음, 분산 매우 큼
- REINFORCE: 여전히 unbiased, 분산 감소
- actor-critic: 더 낮은 분산, 대신 critic 근사 오차로 인한 편향 가능

이는 Monte Carlo와 TD의 관계를 정책 최적화 쪽으로 옮긴 것과 같습니다.

## 확률적 정책이 중요한 이유

정책망은 softmax를 통해 행동확률을 출력합니다.

- 확률은 음수가 아님
- 전체 합이 1
- categorical sampling 가능

정책경사에서는 실제 샘플링이 중요합니다. 단순 argmax만 하면 score-function 형태의 확률적 정책미분 구조가 사라집니다.

## 구현 통찰

- 저장된 `prob[action]`는 바로 `pi_theta(a_t | s_t)`입니다.
- actor 손실 쪽으로 critic의 target gradient가 흘러가면 안 되므로 advantage 신호는 detach되어야 합니다.
- actor-critic은 REINFORCE처럼 에피소드 종료까지 기다리지 않고 매 스텝 갱신할 수 있습니다.

## PyTorch 대응 구현

동일한 알고리즘 구조가 다음 PyTorch 버전에도 있습니다.

- [../pytorch/simple_pg.py](../pytorch/simple_pg.py)
- [../pytorch/reinforce.py](../pytorch/reinforce.py)
- [../pytorch/actor_critic.py](../pytorch/actor_critic.py)

프레임워크는 달라도 수학 구조는 동일하므로 비교하며 읽는 것이 좋습니다.

## 더 넓은 맥락

정책경사는 PPO, A2C/A3C, TRPO, SAC 등 현대 강화학습의 중요한 계열로 이어집니다. 이 장은 값기반 사고에서 “행동분포 직접 최적화”로 넘어가는 분기점입니다.
