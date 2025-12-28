---
layout: distill
title: "Serving LLaMA 3-70B on TPUs"
# permalink: /main/
description: "TPU v5e에서 LLaMA 3-70B 모델을 서빙하는 방법을 자세히 살펴보겠습니다. roofline에서 서빙하는 데 비용이 얼마나 들까요? KV 캐시의 크기는 얼마일까요? 어떤 배치 크기를 사용해야 할까요? 추론 중 파라미터와 활성화는 어떻게 샤딩될까요? 프로덕션 환경에서의 지연 시간과 처리량에 대한 대략적인 추정치를 계산해 보겠습니다."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 8

previous_section_url: "../inference"
previous_section_name: "Part 7: Inference"

next_section_url: ../profiling
next_section_name: "Part 9: Profiling"

giscus_comments: true

authors:
  - name: Jacob Austin
    url: "https://www.jacobaustin.org/"
    affiliations:
      name: Google DeepMind
  - name: Sholto Douglas
    url: "https://x.com/_sholtodouglas"
  - name: Roy Frostig
    url: "https://cs.stanford.edu/~rfrostig/"
  - name: Anselm Levskaya
    url: "https://anselmlevskaya.com/"
  - name: Charlie Chen
    url: "https://x.com/charliexychen"
  - name: Sharad Vikram
    url: "https://sharadvikram.com/"
  - name: Federico Lebron
    url: "https://fedelebron.com/"
  - name: Peter Choy
    url: "https://x.com/pchoy95"
  - name: Vinay Ramasesh
    url: "https://x.com/vinayramasesh"
  - name: Albert Webson
    url: "https://representation.ai/"
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "What's the LLaMA Serving Story?"
  - subsections:
    - name: "Thinking about throughput"
    - name: "What about prefill?"
  - name: "Visualizing the Latency Throughput Tradeoff"
  - name: "Worked Problems"

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

<p markdown=1 class="takeaway">
<b>번역 안내:</b> 원저자([Jacob Austin](https://www.jacobaustin.org/))의 허락을 받아 원문을 번역 중입니다.<br> 
해당 글의 1인칭은 원문 저자를 지칭합니다.<br> 
원문: [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)<br> 
번역: [신종훈](https://www.linkedin.com/in/michael-shin-3522a6189/)</p>

*이 섹션에서는 LLaMA-3를 서빙하는 데 필요한 것과 얼마나 효율적으로 수행할 수 있는지 살펴볼 것입니다. 이전의 "applied" 섹션과 마찬가지로, 정답을 찾아보기 전에 펜과 종이를 가지고 스스로 답을 찾아보세요!*

## What's the LLaMA Serving Story?

LLaMA 3-70B가 어떻게 생겼는지 상기해 봅시다 ([섹션 6](../applied-training) 참조):

| **hyperparam** | **value** |
| --------------------------- | :-------: |
| $$n_\text{layers}$$ (L)     |    80     |
| $$d_\text{model}$$ (D)      |   8,192   |
| $$d_{ff}$$ (F)              |  28,672   |
| $$n_\text{heads}$$ (N)      |    64     |
| $$n_\text{kv heads}$$ (K)   |     8     |
| $$d_\text{qkv}$$ (H)        |    128    |
| $$n_\text{embeddings}$$ (V) |  128,256  |

간단한 질문으로 시작해 봅시다: **어떤 하드웨어에서 서빙해야 할까요?** 정답은 기본적으로 FLOPs / dollar가 가장 저렴한 것입니다.<d-footnote>항상 그런 것은 아니며, 때로는 FLOPs보다 더 많은 HBM이나 ICI 대역폭이 중요할 수 있지만, 이는 좋은 휴리스틱입니다.</d-footnote> 이러한 이유로, 우리는 일반적으로 현재 전용 추론 칩인 TPU v5e에서 서빙하기를 원합니다 (비용은 2025년 2월 기준 [Google Cloud 가격 책정](https://cloud.google.com/tpu/pricing)에서 가져옴):

| **TPU type** | **bfloat16 FLOPs/s** | **Google Cloud USD / hour** | **FLOPs / $** |
| ------------ | :------------------: | :-------------------------: | :-----------: |
| H100         |       9.9e14         |           $10.8             |    3.3e17     |
| v5p          |      4.59e14         |            $4.2             |    3.9e17     |
| v5e          |      1.97e14         |            $1.2             |  **5.8e17** |

각 TPU v5e는 16GB의 HBM을 가지고 있어 모델을 상당히 공격적으로 샤딩해야 합니다. 우리에게 중요할 수 있는 몇 가지 기본 수량을 생각하는 것부터 시작해 봅시다:

**Question:** LLaMA 3-70B의 토큰당 KV 캐시는 얼마나 큰가요? *int8로 저장한다고 가정할 수 있습니다. 이는 주어진 토폴로지에서 배치 크기가 얼마나 클 수 있는지를 결정합니다.*

{% details 다 생각해 보셨다면 여기를 클릭하세요! %}

LLaMA 3-70B는 8개의 KV 헤드를 가지고 있으므로 토큰당 크기는 `2 * K * H * L = 2 * 8 * 128 * 80 = 160kB`입니다.

**이것이 얼마나 큰지 주목하세요!** 시퀀스 길이가 32k 토큰인 경우(일반적임) `162e3 * 32,768 = 5.3GB / sequence`를 사용합니다. BS=240의 경우 이는 1.3TB입니다! TPU v5e는 각각 16GB만 가지고 있으므로 이만큼의 메모리를 맞추려면 약 `(70e9 + 1.3e12) / 16e9 = 86`개의 TPU v5e 칩이 필요합니다. 또한 이것이 70GB의 모델 파라미터에 비해 얼마나 큰지 주목하세요.

{% enddetails %}

**Question:** L3 70B를 배치 크기 32, 시퀀스 길이 8192, 모든 것(파라미터 및 KV)을 int8로 서빙하고 싶다고 가정해 봅시다. 총 메모리는 얼마나 사용될까요? 이것을 서빙할 수 있는 가장 작은 슬라이스는 무엇인가요?

{% details Answer %}

KV는 int8에서 `160e3` 바이트이므로 총 KV 메모리는 `160e3 * 8192 * 32 = 41.9e9` 바이트입니다. 파라미터당 1바이트이므로 파라미터는 `70e9` 바이트입니다. 따라서 총 메모리 사용량은 `41.9e9 + 70e9 = 112GB`입니다.

사용할 수 있는 가장 작은 슬라이스는 `112e9 / 16e9 = 7`개의 TPU, 즉 (짝수 크기로 반올림하여) TPU v5e `4x2`가 될 것입니다. 이것은 빡빡할 수 있으며 다른 오버헤드를 고려할 때 딱 맞지 않을 수도 있으므로 최소한 `4x4`가 필요할 수 있습니다(또는 배치 크기를 줄여야 함).

{% enddetails %}

**Question:** TPU v5e `4x2`에서 이 배치 크기와 양자화로, 디코드 단계당 대략 어떤 지연 시간을 예상할 수 있나요? 처리량(tokens / sec / chip)은 어떤가요? `4x4`는 어떨까요? *FLOPs는 bfloat16에서 수행되고 모든 것이 완전히 샤딩되었다고 가정합니다.*

{% details Answer %}

이전 섹션의 공식을 호출할 수 있습니다.

$$\begin{align*}
\tiny \text{Theoretical Step Time (General)} = \underbrace{\frac{\text{Batch Size} \times \text{KV Cache Size}}{\tiny \text{Total Memory Bandwidth}}}_{\text{Attention (always bandwidth-bound)}} + \underbrace{\max\left(\frac{2 \times \text{Batch Size} \times \text{Parameter Count}}{\text{Total FLOPs/s}}, \frac{\text{Parameter Size}}{\text{Total Memory Bandwidth}}\right)}_{\tiny \text{MLP (can be compute-bound)}}
\end{align*}$$

여기서 파라미터는 int8이지만 FLOPs는 bfloat16이므로 임계 배치 크기는 약 120이 될 것입니다. RHS 최대값을 수동으로 계산할 수도 있지만, 이는 기본적으로 우리가 이미 여러 번 수행한 계산입니다. **따라서 우리는 matmul과 FLOPs 모두에 대해 memory-bound 영역에 잘 들어와 있습니다.**

엄밀히 메모리 대역폭만 보면, 단계 시간은 기본적으로 `(KV size + param size) / (8 * HBM bandwidth) = 112e9 / (8 * 8.1e11) = 17ms`입니다. **따라서 이론적으로 단계 시간은 약 17ms입니다.** 처리량은 `32 / .017 = 1882 tokens / sec` 또는 `1882 / 8 = 235 tokens / sec / chip`입니다.

여기서 한 가지 주의할 점은 matmul에서 ICI bound가 될 수 있는지 확인하는 것입니다. 여기서 2개의 축을 할당할 수 있으므로 이론적으로 $Y > 2 * F / 2200 = 2 * 28672 / 2200 = 26$일 때 ICI bound인데, 우리는 괜찮습니다!

`4x4`에서 실행하더라도 ICI 측면에서 여전히 괜찮으므로 지연 시간은 `17 / 2 = 8.5ms`로 떨어지지만 칩당 처리량은 동일하게 유지됩니다.

{% enddetails %}

### Thinking about throughput

처리량에 대해서만 생각하는 시간을 가져봅시다. 처리량을 최적화할 때 우리는 compute bound가 되기를 원합니다. 즉, 모든 TPU MXU 용량을 활용하는 데 가까워지는 것입니다. 일반적으로 이는 배치 크기를 가능한 한 크게 하여 가능한 한 많은 작업을 수행하고자 함을 의미합니다.

**Question:** TPU v5e에서 bfloat16 가중치와 활성화를 사용할 때, matmul에서 compute-bound가 되려면 배치 크기가 얼마나 커야 하나요? int8 가중치를 사용하지만 FLOPs는 bfloat16으로 수행하면 어떨까요? int8 가중치와 int8 FLOPs는 어떨까요?

{% details Answer %}

섹션 7에서 논의했듯이, $B \ll D, F$인 bfloat16 matmul의 경우 다음이 성립합니다.

$$\begin{equation*}
T_\text{math} > T_\text{comms} \leftrightarrow \frac{2BDF}{2DF} \geq \frac{\text{TPU bfloat16 FLOPs/s}}{\text{HBM bandwidth}} = 240
\end{equation*}$$

가중치가 int8일 때 분모에서 2배를 잃게 되므로 $2BDF / DF = 2B > 240$, 또는 동일하게 $B > 120$이 되며, 이는 이전 임계 배치 크기의 절반입니다. 이것은 우리에게 정말 도움이 됩니다! int8 가중치와 int8 FLOPs를 수행할 때 TPU FLOPs/s에 대해 int8 값을 사용해야 하는데, 이는 bfloat16의 1.97e14에서 거의 두 배인 3.94e14로 증가합니다. 즉, 우리는 다시 약 $B > 240$으로 돌아갑니다.

int8 가중치와 bfloat16 FLOPs의 경우는 꽤 흔합니다. 파라미터를 손실 없이 양자화하는 것이 저정밀도 산술을 수행하는 것보다 종종 더 쉽기 때문입니다.

{% enddetails %}

**Question:** 8k 컨텍스트로 bfloat16, int8, int4(KV 및 파라미터 모두)를 사용하여 LLaMA 3-70B를 서빙할 수 있는 가장 작은 TPU v5e 토폴로지는 무엇인가요? *이 문제에서 KV 캐시는 무시할 수 있을 만큼 작다고 생각할 수 있습니다.*

{% details Answer %}

쉽습니다! 아주 작은 배치 크기에 괜찮다면 유일한 제한은 HBM에 파라미터 메모리를 맞추는 것입니다. 즉, `ceil(num_params * sizeof(dtype) / HBM per TPU`, 또는 `ceil(70e9 * sizeof(dtype) / 16e9)`를 가장 가까운 합리적인 토폴로지(2의 배수)로 반올림한 것입니다:

| dtype | param size | KV size / token (bytes) | min TPU v5es | actual min slice | remaining HBM for KV caches | num KV caches @ 8k |
| :---: | :--------: | :---------------------: | :----------: | :--------------: | :-------------------------: | :----------------: |
| bf16  |   140GB    |          324kB          |     8.75     |  4x4 = 16 chips  |             116             |         43         |
| int8  |    70GB    |          162kB          |     4.38     |  4x2 = 8 chips   |             58              |         43         |
| int4  |    35GB    |          81kB           |     2.81     |  2x2 = 4 chips   |             29              |         43         |
| int8  |    70GB    |          162kB          |     4.38     |  4x2 = 8 chips   |             58              |         43         |
| int4  |    35GB    |          81kB           |     2.81     |  2x2 = 4 chips   |             29              |         43         |

정말 멋집니다! 원한다면 LLaMA 70B를 TPU v5e 2x2에 맞출 수 있다는 것을 말해줍니다. 단, KV 캐시의 수가 매우 적다는 것을 알 수 있습니다. 그것이 바로 우리의 배치 크기입니다! 즉, 우리는 끔찍한 FLOPs 활용률을 얻게 될 것입니다. 배치 크기를 240까지 올리기 위해 더 큰 토폴로지를 사용하는 것이 훨씬 더 기쁠 것입니다.

{% enddetails %}

**Question:** 이 토폴로지들에 맞는 가장 큰 배치 크기를 사용한다고 가정할 때, 각 generate 단계에 대해 어떤 지연 시간을 예상할 수 있을까요?

{% details Answer %}

HBM을 가득 채울 배치 크기를 선택하고 있기 때문에 이것 또한 쉽습니다! 이것은 단지 전체 TPU v5e 분량의 바이트를 MXU로 로드하는 데 얼마나 걸리는지의 문제입니다. 이것은 단지 `v5e HBM / v5e HBM memory bandwidth = 16GB / 8.2e11 = 19ms`이므로 **19ms / step**입니다. 생성의 중앙값 길이가 512 토큰이라고 가정하면 각 디코딩에 약 9초가 걸립니다. 더 작은 배치 크기로 약간 더 나은 지연 시간을 얻을 수 있습니다. 예를 들어 int4의 모델 파라미터만 본다면 HBM이 더 이상 꽉 차 있지 않으므로 최소 지연 시간은 약 10ms / step입니다.

{% enddetails %}

<p markdown=1 class="takeaway">**Takeaway**: HBM에서 MXU로 모든 모델 파라미터를 로드하는 데 걸리는 시간을 물어봄으로써 항상 디코드 지연 시간의 하한을 정할 수 있습니다. KV 캐시가 작을 때, 각 레이어를 가중치를 청크 단위로 로드하고 폐기하는 것으로 생각할 수 있습니다. 큰 배치 크기나 많은 기기 간 통신을 사용하지 않는 한, 이는 종종 합리적인 경계입니다(1.5배 이내). 배치 크기가 클 때는 KV 캐시 로딩이 파라미터를 지배하므로 KV 캐시 로딩도 모델링해야 합니다.</p>

마찬가지로 FLOPs-bound 영역(예: 훈련 또는 대규모 배치 추론)에서는 통신이 없다고 가정하는 $$\text{Total FLOPs} / (N \cdot C) = 2 \cdot \text{param count} \cdot B / (N \cdot C)$$ 하한을 사용할 수 있습니다.

**Question:** 이들 각각에 대해, 이것이 우리에게 주는 칩당 처리량은 얼마인가요(queries / chip 측면에서)? *중앙값 디코드 길이는 512 토큰이라고 가정할 수 있습니다.*

{% details Answer %}

이것은 중요한 질문입니다. 비용 / 토큰과 정확히 상관관계가 있기 때문입니다.

중앙값 디코드 길이에 대한 가정하에 처리량은 단지 $$B / (\text{per-step latency} \cdot \text{median steps} \cdot N) \approxeq 43 / (0.019 * 512 * N)$$입니다. 이는 대략 $$(4.42 / N)$$QPS를 제공하므로$$N$$을 대입하면 다음과 같습니다:

|  dtype   | QPS / chip |
| :------: | :--------: |
| bfloat16 |    0.27    |
|   int8   |    0.55    |
|   int4   |    1.11    |
|   int8   |    0.55    |
|   int4   |    1.11    |

이것은 순방향 패스의 작업 메모리(활성화 및 어텐션에 할당된 메모리)를 완전히 무시하므로 다소 낙관적입니다. Flash Attention을 사용하면 터무니없지는 않지만 현실적이지도 않습니다. 실제 숫자는 아마도 이것의 1/2 정도일 것입니다. 절대적인 최대 처리량을 위해서는 아마도 칩 수를 두 배 이상 늘리고 배치 크기도 크게 늘리고 싶을 것입니다.

{% enddetails %}

**Question:** 위의 각 예에 대해 토폴로지를 두 배로 늘리면 피크 처리량은 어떻게 변할까요?

{% details Answer %}

bfloat16에서 4x8 슬라이스를 사용하면 KV 캐시를 위해 372GB가 남게 되며, 이를 통해 배치 크기를 140까지 늘릴 수 있습니다. 그러면 단계 시간은 동일하게 유지되므로 `14.39 / num_chips`의 처리량을 갖게 되며, 이는 다음과 같습니다.

|        dtype        | QPS / chip |
| :---------------: | :--------: |
| bfloat16 (on 4x8) |    0.44    |
|   int8 (on 4x4)   |    0.90    |
|   int4 (on 2x4)   |    1.80    |

더 늘리면 더 큰 승리를 얻을 수 있습니다! 큰 결론은 KV 캐시 크기에 의해 제한되는 경우 **가장 작은 토폴로지가 모든 경우에 가장 성능이 좋은 토폴로지는 아니라는 것**입니다.

{% enddetails %}

**Question:** 이제 샤딩 문제에 대해 파고들어 봅시다. TPU v5e 4x8에서 bfloat16으로 서빙하고 싶다고 가정해 봅시다. generation 중 TPU v5e 4x8에서 모델에 어떤 샤딩을 사용하시겠습니까? comms bound가 되는 것을 피할 수 있을까요?

{% details Answer %}

이전 섹션에서 논의했듯이 generation 중에는 실제로 샤딩을 위한 하나의 옵션만 있습니다: 모델 병렬 처리. comms bound가 되기 전까지 얼마나 많이 할 수 있을까요? 이전 섹션에서 논의했듯이 모델은 대략 다음과 같을 때 comms bound가 됩니다.

$$Y > \frac{F \cdot M_Y}{2200}$$
$$Y > \frac{F \cdot M_Y}{2200}$$

LLaMA 3-70B의 경우 `F = 28,672`이므로, 2축의 모델 샤딩을 수행하면 대략 $$Y = 28672 \cdot 2 / 2200 = 26$$을 얻으므로 일반적으로 comms bound 없이 약 16개의 칩까지 확장할 수 있으며, 이는 `4x4`를 사용할 수 있게 해주지만 `4x8`은 아닙니다. 일반적으로 계산을 완벽하게 중첩하지 않으므로 이 추정치조차 지나치게 낙관적입니다.

**Takeaway: 순수 모델 병렬 처리로는 4x8에서 실제로 서빙할 수 없습니다.** 여기서 할 수 있는 최선은 4x2 또는 _아마도_ 4x4입니다.

그러나 논의했듯이 배치 크기가 작을 때 모델은 FLOPs bound가 아니라 memory-bandwidth-bound이므로 처리량에 큰 해를 끼치지 않고 종종 더 많은 모델 병렬 처리를 수행할 수 있습니다. 우리는 이전에 이 값이 대략 $Y=F / (8\cdot B)$라고 말했으므로, 배치 크기 64를 수행하면 이론적으로 ICI-bound가 되기 전에 최대 `Y = 28,672 / (8 * 64) = 56`방향 모델 병렬 처리까지 갈 수 있습니다. 이를 확인하기 위해 단일 matmul에 대해 $T_\text{ici comms}$, $T_\text{hbm comms}$, $T_\text{math}$를 살펴볼 수 있습니다. 분명히 다음을 얻습니다:

$$\begin{align*}T_\text{ici comms} = \frac{2BD}{W_\text{ici}} && T_\text{hbm comms} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{math} = \frac{2BDF}{Y \cdot C}\end{align*}$$

`4x8`의 경우 $T_\text{ici comms}$ = `(2 * 64 * 8192) / 9e10 = 11us`, $T_\text{hbm comms}$ = `(2 * 8192 * 28,672) / (32 * 8.1e11) = 18us`, $T_\text{math}$ = `(2 * 64 * 8192 * 28,672) / (32 * 1.97e14) = 4us`가 되므로 이론적으로 우리는 여전히 HBM 대역폭 제한 상태이며, 이는 훌륭합니다! *참고로 `4x4`에서 `4x8`로 확장하는 것은 처리량 관점에서는 도움이 되지 않을 수 있지만 지연 시간은 줄어들 것입니다!

int8 및 int4 구성을 보면 순수 모델 병렬 처리로 *수행할 수 있습니다*. 따라서 우리는 양자화가 실제로 더 빠른 FLOPs 이상의 의미 있는 이점을 제공하는 지점에 도달했습니다: comms-bound가 되기 전에 더 큰 배치 크기를 사용할 수 있게 해줍니다. **따라서 이 이야기의 결론은 4x8에서 최대 처리량을 달성할 수는 없지만 int8 및 int4 구성의 경우 순수 모델 병렬 처리를 수행할 수 있다는 것입니다**.

{% enddetails %}

<p markdown=1 class="takeaway">**Tip**: 유용한 모델 병렬 처리의 최대량은 $$d_{ff}$$와 모델을 샤딩하는 축의 수에 따라 다릅니다. 최대값은 일반적으로 모델 크기에 따라 8에서 32 사이입니다. 이 한계를 넘어서 확장하여 약간의 처리량 비용으로 지연 시간을 개선할 수 있습니다.</p>

### What about prefill?

prefill은 훨씬 간단하기 때문에 여기서 대부분 무시했습니다. 몇 가지 개념을 합쳐서 엔드투엔드 그림을 생각해 봅시다.

**Question:** prefill 중 40% FLOPs 활용률을 달성한다고 가정해 봅시다. 16개의 TPU v5e 칩에서 길이 8192의 prefill은 얼마나 걸릴까요?

{% details Answer %}

8k 토큰에서 우리는 확실히 compute bound이므로 FLOPs에 대해서만 생각하면 됩니다. 모델이 `70e9` 파라미터를 가지고 있으므로 각 순방향 패스는 `2 * 70e9 * B` FLOPs를 사용한다는 것을 알고 있습니다. 40% MFU(FLOPs 활용률)를 가정하면, 이는 약 `2 * 70e9 * 8192 / (16 * 1.97e14 * 0.4) = 0.91s`의 실행 시간을 제공합니다. 이전에 살펴보았던 숫자들과 비교하면 실제로는 꽤 깁니다!

{% enddetails %}

**Question:** 중앙값 prefill 길이가 8192 토큰이고 중앙값 decode 길이가 4096 토큰이라고 가정해 봅시다. generate 배치 크기가 32라고 가정합니다. 평균적으로 단계당 얼마나 많은 시퀀스가 디코딩을 완료합니까? 평균적으로 매 단계마다 얼마나 많은 토큰이 KV 캐시에서 퇴출됩니까?

{% details Answer %}

이것은 일종의 간단한 문제입니다. 중앙값 decode 길이가 4096 토큰이므로 시퀀스는 대략 1 / 4096 토큰마다 완료됩니다. 배치 크기가 32인 경우 단계당 `32 / 4096`개의 시퀀스가 퇴출됩니다. KV 캐시 길이는 대략 `8192 + 4096`이므로, 이는 단계당 `32 * (8192 + 4096) / 4096 = 96`개의 토큰이 퇴출됨을 의미합니다. 일반 공식은 $B * (P + G) / G$이며 여기서 $P$와 $G$는 prefill 및 generate 길이입니다.

{% enddetails %}

**Question:** 중앙값 prefill 길이가 8192이고 중앙값 decode 길이가 512인 분산형 서빙을 수행한다고 가정해 봅시다. bfloat16에서 위에서 계산된 prefill 및 generate 지연 시간을 가정합니다. 둘 다 완전히 포화 상태로 유지하려면 prefill:generate 서버의 비율이 어떻게 되어야 할까요?

{% details Answer %}

이것은 꽤 재미있는 질문입니다. $P$를 prefill 서버 수, $G$를 generate 서버 수라고 합시다. 따라서 일반적으로 말해서, 이것은 시퀀스를 `P / prefill_latency` 속도로 공급하고 `B * G / (generate_latency * median_decode_length)` 속도로 소비하는 파이프라인 문제입니다. 배치 크기 43(32라고 부릅시다)에서 prefill 단계당 `910ms` 및 decode 단계당 `19ms`를 계산했습니다. 따라서 `P / 0.91 = 32 * G / (0.019 * 512)` 또는 `P = 3G`, 즉 생성 서버보다 약 3배 더 많은 prefill 서버가 필요합니다!

{% enddetails %}

## Visualizing the Latency Throughput Tradeoff

잠시 LLaMA 70B를 고수하며 generation 중 다양한 배치 크기에 대한 지연 시간과 처리량을 실제로 살펴보겠습니다. 이전 섹션에서 PaLM 모델에 대해 보여주었듯이 이것은 처리량/지연 시간에 대한 파레토 프런티어를 제공합니다. MLP 블록에서 compute-bound를 유지하면서 사용할 수 있는 합리적인 한계이므로 16방향 텐서 병렬 처리를 가정해 보겠습니다. 여기서는 TPU v5e 4x4 토폴로지를 사용합니다. **슬라이더는 시퀀스 길이를 제어하므로 더 큰 KV 캐시의 효과를 볼 수 있습니다.**

<div class="l-page">
  <iframe src="{{ 'assets/plotly/pareto.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

* **비용과 지연 시간 간의 트레이드오프가 얼마나 극적인지 확인하세요.** 토큰당 지연 시간을 두 배로 늘리는 비용으로 토큰당 비용을 대략 100배 줄일 수 있습니다. 또한 지연 시간은 낮은 배치 크기에서 5.5ms부터 매우 큰 배치에서 20ms까지 다양할 수 있습니다.
* 2k 컨텍스트에서 처리량이 BS 120 루프라인(int8 가중치이지만 bf16 FLOPs를 수행하므로 여기서는 120)에 도달할 때 약 1 token / ms / chip에서 어떻게 효과적으로 정체되는지 주목하세요. 그러나 시퀀스 길이가 증가함에 따라 이 배치 크기를 메모리에 맞출 수 없으므로 완전 포화 지점에 도달하지 못합니다.
* 동일한 처리량에 대해 큰 배치 크기에서 지연 시간이 얼마나 더 높은지 주목하세요. KV 로딩이 (파라미터 로딩 대신) 지배적이 되기 때문입니다.

비용과 지연 시간의 원인을 파라미터 로딩 시간, KV 로딩 시간, FLOPs 시간으로 나누어 더 잘 이해할 수 있습니다. 빨간색 섹터는 MLP 블록에서 compute-bound가 될 것으로 예상되는 영역입니다.

<div class="l-page">
  <iframe src="{{ 'assets/plotly/latency_breakdown_log.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

이것은 꽤 많은 이야기를 해줍니다. 처음에 파라미터 로딩이 지연 시간의 대부분을 차지하다가 배치 크기가 충분히 커지면 FLOPs와 KV 로딩이 더 중요해지는 것을 볼 수 있습니다. 특히 2048보다 큰 모든 시퀀스 길이에서 FLOPs보다 KV 캐시 로딩에 더 많은 시간을 소비합니다! **따라서 배치 크기를 늘려 하드웨어 활용률을 개선할 수 있지만, 긴 컨텍스트 길이에서는 KV 로딩이 항상 총 단계 시간을 지배합니다.**

<p markdown=1 class="takeaway">**Takeaway:** LLaMA 3-70B의 경우 거의 모든 구성에서 강력하게 KV 캐시 메모리 대역폭 제한(및 HBM 제한) 상태이며, 이는 generation 처리량을 위해 KV 캐시 크기를 줄이는 것이 얼마나 중요한지 강조합니다. 또한 여기서 지연 시간/처리량 트레이드오프가 얼마나 극적인지 주목하세요.</p>

{% details 코드는 꽤 간단합니다. %}

다음은 이러한 루프라인을 계산하는 코드입니다:

```py
import numpy as np

num_chips = 16  # we fix 16 as the amount of total model parallelism we do
param_size = 70e9  # int8 means 1 byte per param
sequence_length = 8192  # can vary this

hbm_bandwidth = 8.20E+11  # v5e
flops = 1.97E+14  # v5e

param_size = bytes_per_param * param_count

def kv_cache_size(bs):
    return 2 * bs * 128 * 8 * 80


def min_topology(bytes):
    return 2 ** np.ceil(np.log2(bytes / 16e9))

def get_max_batch_size(max_num_chips: int = 16):
  # for num_chips in topo_sizes:
  batch_sizes = np.arange(1, 1024, 4)
  kv_sizes = kv_cache_size(sequence_length * batch_sizes)
  num_chips = min_topology(kv_sizes + param_size)
  max_idx = np.where(num_chips <= max_num_chips)[0][-1]
  return max_idx

max_idx = get_max_batch_size(num_chips, sequence_length, param_size)  # get the largest batch size that can fit
batch_sizes = np.arange(1, 512, 1)[:max_idx]
kv_sizes = kv_cache_size(sequence_length * batch_sizes)

kv_comms_time = kv_sizes / (num_chips * hbm_bandwidth)

param_comms_time = param_size / (num_chips * hbm_bandwidth)
param_comms_time = np.asarray([param_comms_time] * batch_sizes.shape[0])

flops_time = 2 * param_count * batch_sizes / (num_chips * flops)  # roughly true in a 2ND sense

mlp_time = np.maximum(flops_time, param_comms_time)
attn_time = kv_comms_time  # always bandwidth-bound for generate

latency = 1000 * (mlp_time + attn_time)
throughput = batch_sizes / (latency * num_chips)
```

우리가 지연 시간을 KV 로딩과 파라미터 로딩이라는 두 가지 소스로 매우 명시적으로 나누고, 지연 시간이 FLOPs 또는 comms 중 더 큰 것에 의해 제한되는 방식을 주목하세요.

{% enddetails %}

## Worked Problems

다음은 몇 가지 풀이 문제입니다. 이 중 일부는 위에서 다룬 내용을 반복하지만 교육적으로 유용할 수 있습니다.

**Question 1:** LLaMA 3-405B에 대한 각 순방향 패스는 토큰당 몇 FLOPs를 사용하나요? FLOPs bound라고 가정할 때 TPU v5e의 N개 칩에서 단일 순방향 패스의 하한은 얼마인가요? comms bound라면 어떨까요? *모델이 단일 칩에 맞지 않는다는 사실은 무시하세요.*

**Question 2:** int8 가중치와 int8 KV 캐시를 사용하여 BS240으로 LLaMA 3-8B를 서빙하고 싶다고 가정해 봅시다. (a) 모델 파라미터 (b) KV 캐시 (c) 피크 작업 활성화(대략)에 사용되는 바이트 수는 얼마인가요? 이것을 실행할 수 있는 가장 작은 토폴로지는 무엇인가요?

**Question 3:** TPU v5e에서 LLaMA 3-405B를 어떻게 서빙하시겠습니까? int8 가중치와 bfloat16 FLOPs를 가정합니다. 15ms / token의 확고한 제한이 있다고 가정할 때 달성할 수 있는 가장 높은 처리량 구성은 무엇인가요? 이론적 최소 단계 시간은 얼마인가요?

<h3 markdown=1 class="next-section">섹션 8은 여기까지입니다! XLA 및 TPU 프로파일링에 대해 자세히 알아보는 섹션 9를 보려면 [여기](../profiling)를 클릭하세요.</h3>