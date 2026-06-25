# Stage 1 WDP 기반 5PL 다품종 SKU 거래 매칭 모델

본 문서는 `지윤 model2.ipynb`와 `stage1_wdp_trading_matching_T1_AM.lp`를 기준으로 정리한 2차년도 5PL 플랫폼 매칭 모델 설명이다. 
기존 모델은 순차적으로 주문 하나를 선택해 처리하는 구조를 중심으로 설명했다면, 
README2는 여러 구매자 주문을 동시에 고려하는 Stage 1 winner determination problem(WDP)형 trading matching 구조에 초점을 둔다.

## 모델의 핵심 방향

본 모델은 다수 화주가 보유한 다품종 SKU 재고와 다수 구매자의 multi-SKU 주문을 연결하는 5PL 플랫폼의 1단계 거래 매칭 문제를 다룬다. 플랫폼은 각 주문을 수락할지, 수락한다면 어떤 화주의 어떤 SKU lot을 얼마나 사용할지 결정한다.

핵심 의사결정은 다음과 같다.

- 어떤 주문을 수락할 것인가
- 수락된 주문의 SKU별 수요를 얼마나 충족할 것인가
- 어떤 화주-SKU lot을 사용할 것인가
- 화주별 재고와 최소 lot 거래조건을 만족할 것인가
- 구매자-화주 연결에 따른 배송/거래 고정비를 감안해 전체 surplus가 양수인가

따라서 본 모델은 단순 가격결정 모형이라기보다, 여러 주문과 여러 화주 재고를 동시에 놓고 플랫폼이 거래 조합을 선택하는 WDP형 allocation model에 가깝다.

## 1차년도 모형 대비 차별점

1차년도 모형이 가격결정과 거래 가능성 판단에 초점을 두었다면, 본 모델은 실제 플랫폼 운영에서 필요한 매칭과 이행 가능성 판단을 더 구체적으로 반영한다.

주요 차별점은 다음과 같다.

- 여러 주문을 동시에 고려하여 수락 여부를 결정한다.
- 주문별 multi-SKU 수요와 SKU별 최소 충족률을 반영한다.
- 화주별 SKU 재고와 최소 거래비율을 반영한다.
- 주문-SKU-화주 조합 단위로 이행 수량을 결정한다.
- 구매자-화주 연결 여부에 따른 고정 배송/거래 비용을 목적식에 포함한다.
- 거래 가능성뿐 아니라 실제 이행 수량과 사용 lot을 함께 결정한다.
- 매칭 이후 가격정산은 별도 surplus sharing rule로 계산할 수 있다.

요약하면, 본 모델은 가격결정 중심의 정적 모형에서 여러 주문, 다품종 SKU, 화주 lot, 배송 연결비용을 함께 고려하는 1단계 WDP 기반 플랫폼 운영 모형으로 확장한 형태이다.

## 집합과 인덱스

- `O`: 구매자 주문 집합
- `B`: 구매자 집합
- `S`: 화주 집합
- `P`: SKU 집합

LP 파일에서는 다음과 같은 인덱스 구조가 나타난다.

```text
q[O1,S1,P1]
q[O2,S2,P3]
q[O5,S4,P4]
```

즉, `q[o,s,p]`는 주문 `o`의 SKU `p`를 화주 `s`의 재고로 이행하는 수량이다.

## 주요 파라미터

- `D_op`: 주문 `o`에서 SKU `p`의 요청 수량
- `alpha_op`: 주문 `o`의 SKU `p`에 대한 최소 충족률
- `I_sp`: 화주 `s`가 보유한 SKU `p`의 가용 재고
- `a_sp`: 화주 `s`가 SKU `p`에 대해 요구하는 최소 수용가격
- `b_op`: 주문 `o`의 SKU `p`에 대한 구매자 최대 지불의향
- `beta_sp`: 화주 `s`의 SKU `p` lot 최소 거래비율
- `B_o`: 주문 `o`의 예산 상한
- `F_bs`: 구매자 `b`와 화주 `s`가 연결될 때 발생하는 추정 배송/거래 고정비

## 의사결정변수

- `x_order[o]`: 주문 `o` 수락 여부
- `q[o,s,p]`: 주문 `o`의 SKU `p`를 화주 `s`로부터 이행하는 수량
- `Q[o,p]`: 주문 `o`의 SKU `p` 총 이행 수량
- `z[o,s,p]`: 주문 `o`-화주 `s`-SKU `p` 조합 사용 여부
- `y_seller_sku[s,p]`: 화주 `s`의 SKU `p` lot 사용 여부
- `w_buyer_seller[b,s]`: 구매자 `b`와 화주 `s`의 거래 연결 발생 여부

`z`는 세부 SKU 배정 여부를 나타내고, `w_buyer_seller`는 한 구매자가 특정 화주와 연결되었는지를 나타낸다. 이를 통해 같은 화주가 여러 SKU를 공급할 때 고정 배송비를 한 번만 반영할 수 있다.

## 목적식

목적식은 대략 다음 구조를 가진다.


max  sum_o sum_s sum_p unit_surplus_osp * q[o,s,p]
     - sum_b sum_s F_bs * w_buyer_seller[b,s]


LP 파일에서는 다음처럼 나타난다.


Maximize
  1970 q[O1,S1,P1] + ... + 2240 q[O5,S4,P4]
  - 3000 w_buyer_seller[B1,S1] - ... - 5000 w_buyer_seller[B4,S4]


이는 SKU별 거래 수량에서 발생하는 단위 surplus를 최대화하되, 구매자-화주 연결에 따른 배송/거래 고정비를 차감하는 구조다. 따라서 모델은 단순히 단위마진이 큰 조합을 많이 선택하는 것이 아니라, 화주를 추가로 연결할 때 발생하는 고정비까지 고려한다.

## 핵심 제약식

### 1. SKU별 총 이행 수량 정의

Q[o,p] = sum_s q[o,s,p]


각 주문-SKU의 총 이행량은 여러 화주가 공급한 수량의 합으로 정의된다.

### 2. 최소 충족률과 수요 상한


Q[o,p] >= alpha_op * D_op * x_order[o]
Q[o,p] <= D_op * x_order[o]


주문이 수락되면 SKU별 최소 충족률 이상을 만족해야 하며, 요청 수량을 초과해서 이행할 수 없다. 주문이 수락되지 않으면 해당 주문의 이행 수량은 0이 된다.

### 3. 화주 재고 제약

sum_o q[o,s,p] <= I_sp

화주 `s`는 자신이 보유한 SKU `p`의 재고 이상으로 공급할 수 없다.

### 4. 조합 사용 여부 연결


q[o,s,p] <= M_osp * z[o,s,p]
z[o,s,p] <= x_order[o]
z[o,s,p] <= y_seller_sku[s,p]


주문이 수락되고 해당 화주-SKU lot이 사용되는 경우에만 실제 수량 배정이 가능하다.

### 5. 화주 SKU lot 최소 거래비율


sum_o q[o,s,p] >= beta_sp * I_sp * y_seller_sku[s,p]
sum_o q[o,s,p] <= I_sp * y_seller_sku[s,p]


화주 `s`의 SKU `p` lot을 사용하기로 했다면, 해당 lot에 대해 최소 거래비율 이상을 거래해야 한다. 반대로 lot을 사용하지 않으면 해당 lot에서 공급되는 수량은 0이다.

### 6. 주문 예산 제약


sum_p b_op * Q[o,p] <= B_o * x_order[o]


현재 LP에서는 주문 예산이 구매자의 SKU별 최대 지불의향 기준으로 제한된다. 향후 실제 정산가격을 예산 제약에 직접 반영하려면 `buyer_payment_price[o,s,p] * q[o,s,p]` 형태로 확장할 수 있다.

### 7. 구매자-화주 연결 비용


z[o,s,p] <= w_buyer_seller[b,s]
w_buyer_seller[b,s] <= sum 관련 z[o,s,p]


특정 구매자의 주문이 화주 `s`의 재고를 하나라도 사용하면 `w_buyer_seller[b,s]`가 1이 된다. 목적식에서 `w_buyer_seller`에 고정비가 부과되므로, 모델은 너무 많은 화주를 쪼개 쓰는 것을 자연스럽게 억제한다.

## 가격정산과 surplus sharing

Stage 1 LP의 목적식은 매칭과 이행 수량을 결정하기 위한 surplus 기반 allocation objective다. 실제 가격정산은 최적화 이후 별도의 settlement rule로 계산할 수 있다.

거래 가능한 조합의 단위 거래 surplus는 다음과 같이 정의한다.


unit_trade_surplus = b_op - a_sp


예상 배송비는 Stage 1 목적식과 적자 방지 제약에서 별도로 반영한다. 거래가 성립하면 `unit_trade_surplus`를 플랫폼, 화주, 수하주가 공유한다. 현재 정산 규칙은 다음과 같다.


platform_surplus_share = 0.2
seller_surplus_share   = 0.4
buyer_surplus_share    = 0.4


이에 따라 실제 정산 단가는 다음과 같이 계산한다.


buyer_payment_price = b_op - buyer_surplus_share * unit_trade_surplus
seller_payment_price = a_sp + seller_surplus_share * unit_trade_surplus
platform_unit_fee = platform_surplus_share * unit_trade_surplus


배송비를 고려한 플랫폼의 추정 순이익은 `platform_unit_fee * quantity - estimated_delivery_cost`로 별도 계산한다. 이 정산 구조는 사회적 후생 총량을 바꾸는 장치라기보다, 이미 발생한 surplus를 플랫폼, 화주, 수하주가 나누어 갖게 함으로써 각 참여자의 재참여 유인을 높이는 장치다.

## 현재 모델의 해석

현재 LP 파일을 기준으로 보면 모델은 다음 성격을 가진다.

- 단일 주문 순차 매칭보다 WDP형 다중 주문 동시 매칭에 가깝다.
- 운송사를 명시적으로 선택하기보다는 구매자-화주 연결비용을 통해 배송비를 반영한다.
- SKU별 부분 충족률, 화주 재고, lot 최소 거래비율이 반영되어 있다.
- `w_buyer_seller` 고정비가 있어서 불필요한 화주 분산을 줄이는 효과가 있다.
- 가격정산과 2:4:4 surplus sharing은 LP 내부 가격변수라기보다 매칭 이후 적용되는 사후 정산 룰이다.
2- `buyer_payment_price`, `seller_payment_price`, `buyer_total_payment`, `seller_total_payment`를 확정 거래 record에 저장한다.
- `run_stage1_over_time_windows`를 통해 여러 시간창을 순차적으로 풀고, 각 시간창 이후 재고를 업데이트할 수 있다.

## 향후 보완 방향

- 운송사 또는 fulfillment option 인덱스 `k`를 다시 포함하여 `(주문, 화주, SKU, 운송사)` 조합 단위로 확장한다.
- 실제 정산가격 기준의 예산 제약을 반영한다.
- 현재의 SKU별 split 제약을 주문 전체 split, 화주 수 제한 등 더 다양한 정책 제약으로 확장한다.
- route, time window, 차량 capacity를 포함한 Stage 2 routing model과 연결한다.
- 과도한 bid 또는 ask에 대해서는 hard cap보다 abnormal price flag 또는 risk penalty를 도입한다.
- Stage 1 매칭 결과를 기반으로 Stage 2 배송 가능성 검증 및 route optimization을 수행한다.

## 관련 파일

```text
2nd/
├─ README.md
├─ README2.md
├─ 지윤 model.ipynb
├─ 지윤 model2.ipynb
└─ stage1_wdp_trading_matching_T1_AM.lp
```

- `지윤 model.ipynb`: 초기 순차 매칭 및 가격정산 초안
- `지윤 model2.ipynb`: Stage 1 WDP형 trading matching 모델
- `stage1_wdp_trading_matching_T1_AM.lp`: Gurobi가 생성한 Stage 1 LP 파일
