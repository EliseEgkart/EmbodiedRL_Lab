# Chapter 7: 신경망과 함수근사

## 왜 표 방식이 한계에 부딪히는가

표 기반 방법은 각 상태 혹은 상태-행동 쌍마다 별도 파라미터를 둡니다. 이는 작은 GridWorld에는 적합하지만 다음 경우엔 곧 막힙니다.

- 상태공간이 큼
- 상태가 연속적임
- 비슷한 상태 간 일반화가 필요함

그래서 값함수를 매개변수화된 함수 `f_theta`로 근사합니다.

## 선형대수 기초

가장 기본 블록은 affine map입니다.

`y = xW + b`

[dezero1.py](./dezero1.py)는 행렬곱 자체를 분리해서 보여줍니다. 모든 dense layer의 핵심이기 때문입니다.

## 최적화 관점

학습은 결국 손실 최소화 문제입니다.

`theta^* = argmin_theta L(theta)`

회귀에서는 보통 평균제곱오차를 씁니다.

`L(theta) = (1 / N) sum_i (y_i - f_theta(x_i))^2`

[dezero3.py](./dezero3.py)는 선형회귀를 통해 이 구조를 보여줍니다.

업데이트는

`theta <- theta - eta * grad_theta L(theta)`

입니다.

## 자동미분

[dezero2.py](./dezero2.py)는 Rosenbrock 함수 최적화를 통해

- 연산 그래프 구성
- backward 호출
- gradient 계산
- 파라미터 갱신

의 흐름을 보여줍니다. 이후 RL에서는 값망과 정책망 모두 이 구조 위에서 학습됩니다.

## 비선형성

affine layer만 여러 개 쌓으면 결국 하나의 affine map과 다르지 않습니다. 그래서 sigmoid, ReLU 같은 비선형 활성함수가 필요합니다.

[dezero4.py](./dezero4.py)는 은닉층 비선형성이 있어야 곡선 함수를 근사할 수 있음을 보여줍니다.

## Q 함수 근사

신경망이 한 상태에 대해 모든 행동가치를 동시에 출력한다고 보면

`Q_theta(s) = [Q_theta(s, a_1), ..., Q_theta(s, a_k)]`

가 됩니다.

[q_learning_nn.py](./q_learning_nn.py)의 타깃은

`y = r + gamma max_{a'} Q_theta(s', a')`

이고 손실은

`L(theta) = (Q_theta(s, a) - y)^2`

입니다.

## one-hot 인코딩의 의미

GridWorld 상태는 좌표이므로 신경망 입력으로 직접 쓰기보다 one-hot 벡터로 바꿉니다. 그러면 각 상태가 독립적인 기저벡터를 가지게 되어, 최소한 표 기반 표현을 손상 없이 신경망으로 옮길 수 있습니다.

## semi-gradient의 중요성

타깃에 `Q(next_state)`가 들어가면 자칫 타깃 경로에도 gradient가 흐를 수 있습니다. 그러나 Bellman 회귀에서는 타깃을 고정된 레이블처럼 다뤄야 하므로 detaching이 필요합니다.

이 점은 [q_learning_nn.py](./q_learning_nn.py)와 이후 DQN 코드에서 매우 중요합니다.

## 구현 통찰

- Chapter 7부터 강화학습은 최적화 문제와 완전히 얽힙니다.
- RL은 정답 레이블이 외부에 주어지는 것이 아니라, 현재 네트워크가 만든 타깃을 다시 학습한다는 점에서 일반 지도학습보다 어렵습니다.
- 네트워크가 모든 행동가치를 한 번에 출력하는 구조는 계산 효율과 파라미터 공유 측면에서 유리합니다.

## 다음 장 연결

이제 신경망 Q-learning을 실제 연속 상태 환경에 적용하면 불안정성이 본격적으로 드러납니다. Chapter 8의 DQN은 그 불안정성을 완화하기 위한 기법들입니다.
