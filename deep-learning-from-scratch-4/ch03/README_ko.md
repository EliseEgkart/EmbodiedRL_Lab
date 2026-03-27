# Chapter 3: Bellman 방정식과 최적성

## 저장소 상태

이 저장소에는 `ch03` 실행 코드가 없지만, Chapter 4 이후 모든 구현을 이해하려면 Bellman 방정식을 먼저 정리해야 합니다.

## 반환값의 재귀 구조

반환값은

`G_t = R_{t+1} + gamma R_{t+2} + gamma^2 R_{t+3} + ...`

이고, 첫 항을 분리하면

`G_t = R_{t+1} + gamma G_{t+1}`

가 됩니다. 강화학습에서 거의 모든 업데이트가 결국 이 재귀를 구현합니다.

## Bellman 기대 방정식: 상태가치

고정 정책 `pi`에 대해

`V^pi(s) = sum_a pi(a | s) sum_{s', r} p(s', r | s, a) [r + gamma V^pi(s')]`

입니다.

의미는 다음과 같습니다.

- 정책이 행동을 고르는 평균
- 환경이 다음 상태와 보상을 내는 평균
- 즉시 보상과 할인된 미래가치의 합

[ch04/policy_eval.py](../ch04/policy_eval.py)가 정확히 이 계산을 반복합니다.

## Bellman 기대 방정식: 행동가치

`Q^pi(s, a) = sum_{s', r} p(s', r | s, a) [r + gamma sum_{a'} pi(a' | s') Q^pi(s', a')]`

이 식은 제어 문제에서 매우 중요합니다. 왜냐하면 최적행동을 비교하려면 상태가치보다 행동가치가 직접적이기 때문입니다.

## Bellman 최적 방정식

최적가치를

`V^*(s) = max_pi V^pi(s)`

`Q^*(s, a) = max_pi Q^pi(s, a)`

라 하면

`V^*(s) = max_a sum_{s', r} p(s', r | s, a) [r + gamma V^*(s')]`

`Q^*(s, a) = sum_{s', r} p(s', r | s, a) [r + gamma max_{a'} Q^*(s', a')]`

를 만족합니다.

정책 평균 `sum_a pi(a|s)`가 `max_a`로 바뀌는 작은 차이가 제어의 핵심입니다.

## 정책개선 정리

어떤 정책 `pi`에 대해

`pi'(s) in argmax_a Q^pi(s, a)`

로 새 정책을 만들면

`V^{pi'}(s) >= V^pi(s)`

가 모든 상태에서 성립합니다. 그래서 평가와 개선을 반복하는 정책 반복이 타당합니다.

## 수축 사상 관점

할인된 유한 MDP에서 Bellman 연산자는 sup norm에서 수축입니다.

`||T V - T W||_inf <= gamma ||V - W||_inf`

그래서 반복 적용하면 고정점으로 수렴합니다. [ch04/dp.py](../ch04/dp.py), [ch04/value_iter.py](../ch04/value_iter.py)의 반복문이 이 사실에 기대고 있습니다.

## 동기식과 제자리 업데이트

- 동기식: 이전 값 전체를 고정한 채 새 값을 계산
- 제자리(in-place): 계산 즉시 덮어씀

제자리 방식은 같은 sweep 안에서 더 최신 정보를 재사용하므로 실제로 더 빨리 수렴하는 경우가 많습니다. [ch04/dp_inplace.py](../ch04/dp_inplace.py)가 그 예입니다.

## 구현 통찰

- 코드에서 `reward + gamma * something`이 보이면 거의 항상 Bellman 구조입니다.
- `max_a`가 있으면 최적성 백업이고, 정책 평균이 있으면 평가 백업입니다.
- `max_s |V_new - V_old|` 같은 수렴 기준은 수축 연산자 이론과 직접 연결됩니다.

## 다음 장 연결

Chapter 4는 이 Bellman 식들을 “모델이 알려진 작은 환경”에서 실제로 계산하는 장입니다. 즉, 추상적인 방정식이 실행 가능한 알고리즘이 되는 첫 단계입니다.
