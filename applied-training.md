---
layout: distill
title: "Training LLaMA 3 on TPUs"
# permalink: /main/
description: "이전 섹션에서 배운 내용을 바탕으로 TPU v5p에서 LLaMA 3 모델을 훈련하는 방법을 자세히 살펴보겠습니다. 모델의 크기는 얼마나 클까요? 다양한 구성에서 훈련 비용은 얼마나 들까요? 샤딩은 어떻게 이루어질까요? 이전 섹션의 내용이 실제 모델에 어떻게 적용되는지 대략적인 추정치를 통해 알아보겠습니다."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 6

previous_section_url: "../training"
previous_section_name: "Part 5: Training"

next_section_url: ../inference
next_section_name: "Part 7: Inference"

bibliography: main.bib

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
  - name: "What does LLaMA 3 look like?"
  - name: "Counting parameters and FLOPs"
  - name: "How to shard LLaMA 3-70B for training"
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

_이 섹션의 목표는 이전 섹션의 결과를 매우 실용적인 문제인 LLaMA 3 모델 패밀리(herd) 훈련에 적용하는 것입니다. 이전 섹션들과 달리, 독자 여러분이 직접 많은 작업을 해보기를 권장합니다. 이러한 이유로 각 섹션의 정답을 숨겨두었으니 먼저 스스로 답해 보시기 바랍니다. 펜을 잡고 직접 손으로 계산해 보세요!_

### What does LLaMA 3 look like?

LLaMA-3 모델 패밀리<d-cite key="llama3"></d-cite>는 3가지 주요 모델을 포함합니다: LLaMA 3 8B, 70B, 그리고 405B. 우리는 주로 70B에 초점을 맞추고, 8B와 405B는 마지막 문제 섹션에서 여러분이 직접 탐구하도록 남겨두겠습니다. 다음은 LLaMA [HuggingFace 페이지](https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/config.json)에서 가져온 LLaMA 3-70B의 아키텍처입니다.

| **hyperparam** | **value** |
| --------------------------- | --------- |
| $$n_\text{layers}$$ (L)     | 80        |
| $$d_\text{model}$$ (D)      | 8,192     |
| $$d_{ff}$$ (F)              | 28,672    |
| $$n_\text{heads}$$ (N)      | 64        |
| $$n_\text{kv_heads}$$ (K)   | 8         |
| $$d_\text{qkv}$$ (H)        | 128       |
| $$n_\text{embeddings}$$ (V) | 128,256   |

이 정보를 찾는 것이 얼마나 쉬운지 강조하기 위해, 매핑과 함께 config 자체를 보여드립니다:

{% include figure.liquid path="assets/img/llama-json.png" class="img-fluid" %}

_다양한 오픈 소스 LLM에 대해 이러한 수치들을 큰 표로 만들어두면, 각 모델이 내린 설계 결정을 빠르게 비교하는 데 유용합니다._

### Counting parameters and FLOPs

**Question:** 이 표를 바탕으로 LLaMA 3-70B의 파라미터 수를 계산할 수 있을까요? 🤫 [섹션 4](../transformers)의 내용을 적용하여 70B가 나오는지 확인해 봅시다!

{% details 정답을 보려면 여기를 클릭하세요 (먼저 생각해 보세요!). %}

| param            | formula                                                                                                                                         | count                                                        |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| FFW params       | d_model * d_ff * 3 (for gelu + out-projection) * n_layers                                                                                       | 8,192 * 8,192 * 3.5 * 3 * 80 = **56.3e9** |
| Vocab params     | 2 (input and output embeddings) * n_embeddings * d_model                                                                                        | 2 * 128,256 * 8,192 = **2.1e9** |
| Attention params | n_layers * [ 2 (for q embedding and concatenated output projection) * d_model * n_heads * d_qkv + 2 (for k and v) * d_model * n_kv_heads * d_qkv] | 80 * (2 * 8,192 * 64 * 128 + 2 * 8,192 * 8 * 128) = **12e9** |
|                  |                                                                                                                                                 | 56.3e9 + 2.1e9 + 12e9 = **70.4e9** |

훌륭합니다! 우리가 예상한 숫자가 나왔습니다. 예상대로 FFW 파라미터가 전체 파라미터 수의 대부분을 차지하지만, Attention도 무시할 수 없는 수준임을 알 수 있습니다.

{% enddetails %}

<p markdown=1 class="takeaway">**Takeaway**: MLP 블록의 3가지 큰 가중치 행렬은 Transformer 내의 다른 모든 배열보다 훨씬 크기 때문에, 모델 메모리나 FLOPs를 추론할 때 다른 파라미터들은 거의 무시해도 무방합니다. LLaMA 3-70B의 경우, 70B 파라미터 중 56B를 차지합니다.</p>

이제 FLOPs를 살펴보겠습니다! *[섹션 4](../transformers)의 훈련에 대한 일반적인 규칙을 기억하세요.*

**Question:** LLaMA-3는 훈련 단계당 토큰별로 얼마나 많은 FLOPs를 수행할까요? _이는 전체 훈련 과정이 얼마나 비쌀지 결정하는 데 도움이 됩니다._

{% details 정답을 보려면 여기를 클릭하세요 (먼저 생각해 보세요!). %}

**Answer**: [섹션 4](../transformers)에서 보았듯이, 토큰당 대략 $$6 \cdot \text{param count}$$ FLOPs를 수행하므로, 여기서는 대략 `6 * 70e9 = 4.2e11` FLOPs / token입니다. 이는 단계당 토큰별로 약 0.5 TFLOP입니다. compute-bound라고 가정할 때, 단일 TPU v5p 칩에서 완벽한 FLOPs 활용을 가정하면 대략 `4.2e11 / 4.59E+14 = 1ms`가 걸립니다.

{% enddetails %}

**Question:** LLaMA 3는 약 15조 토큰으로 훈련되었습니다. 총 FLOPs는 얼마인가요?

{% details 정답을 보려면 여기를 클릭하세요 (먼저 생각해 보세요!). %}

**Answer**: 간단합니다. `4.2e11 * 15e12 = 6.3e24 FLOPs`입니다. 6.3 yottaFLOPs입니다. 엄청난 양이죠! 단일 TPU에서는 `6.3e24 / 4.59E+14 = 435년`이 걸립니다. 이것 또한 엄청난 시간입니다!

{% enddetails %}

**Question:** 16x20x28 = 8960개의 칩으로 구성된 전체 TPU v5p pod에서 훈련하고 싶다고 가정해 봅시다. compute-bound라고 가정할 때, bfloat16에서 40% MFU로 훈련하는 데 얼마나 걸릴까요?

{% details 정답을 보려면 여기를 클릭하세요 (먼저 생각해 보세요!). %}

**Answer**: 각 TPU v5p는 초당 4.59e14 FLOPs를 수행할 수 있다는 것을 알고 있습니다. 40% MFU에서, 이는 약 `T = 6.3e24 / (8960 * 4.59e14 * 0.4) = 3.8e6 초`가 걸립니다. **약 44일입니다!** 40% MFU를 실제로 달성할 수 있다고 가정하면 꽤 합리적입니다.

{% enddetails %}

**Question:** LLaMA 3-70B는 약 4M 토큰의 배치 크기로 사전 훈련되었습니다. 이 배치 크기로 훈련하려면 최소 몇 개의 TPU가 필요할까요? _bfloat16 파라미터와 float32 옵티마이저 상태를 가정하고, 레이어당 4번 그라디언트를 체크포인트한다고 가정합니다._

{% details 정답을 보려면 여기를 클릭하세요 (먼저 생각해 보세요!). %}

**Answer**: 이 질문은 주로 메모리 사용량에 관한 것입니다. 사용 가능한 컴퓨팅에 대한 유일한 엄격한 제약 조건이기 때문입니다. 훈련 중에 HBM의 세 가지 주요 용도는 모델 파라미터, 옵티마이저 상태, 그리고 그라디언트 체크포인트입니다. bfloat16 가중치, float32 옵티마이저 상태, 그리고 _매우_ 보수적인 그라디언트 체크포인팅 방식(레이어당 4회)을 가정하면 다음과 같습니다:

| **Params** | 2 * 70GB                | ~140GB  |
| **Optimizer State** | 8 * 70GB                | ~560GB  |
| **Gradient Checkpoints** | 2 * 8192 * 4e6 * 4 * 80 | ~20.9TB |
| **Total** |                         | ~21.6TB |

여기서 합계는 약 21.6TB입니다. 매우 보수적인 체크포인팅 방식을 사용하더라도 그라디언트 체크포인팅이 메모리 상황을 강력하게 지배한다는 것을 알 수 있습니다. 기술적으로는 레이어당 1개의 체크포인트로 가거나 마이크로배칭을 할 수 있지만, 이는 합리적인 그림입니다. 이러한 가정하에, 각 TPU v5p는 96GB의 HBM을 가지고 있으므로 `21.6e12 / 96e9 = 225` TPU가 필요합니다. 사실 그리 많지 않습니다!

*왜 이렇게 하지 않을까요?* 글쎄요, 훈련하는 데 `44일 * 8960 / 225 = 1752일`이 걸리기 때문입니다. 거의 4년입니다. **너무 깁니다.** 하지만 이것은 우리가 메모리에 묶여서가 아니라 추가적인 FLOPs가 필요해서 이러한 대규모 클러스터를 사용하고 있음을 분명히 보여줍니다.

{% enddetails %}

**Question:** 위 질문과 동일한 가정하에, 8960개의 TPU v5p 칩을 사용한다면 칩당 얼마나 많은 메모리를 사용하게 될까요?

{% details 정답을 보려면 여기를 클릭하세요 (먼저 생각해 보세요!). %}

**Answer**: 총 메모리는 여전히 약 21.6TB이므로, 칩당 약 2.4GB를 사용하게 되는데, 이는 기본적으로 아무것도 아닙니다. 훨씬 더 공격적인 체크포인팅, 예를 들어 레이어당 12개의 체크포인트를 수행하더라도 칩당 8GB에 불과합니다. 이러한 규모의 훈련 중에는 메모리 부족 문제와는 거리가 멉니다.

{% enddetails %}

<p markdown=1 class="takeaway">**Takeaways**: 기술적으로는 매우 작은 토폴로지에서도 매우 큰 모델을 훈련할 수 있지만, 시간이 오래 걸릴 수 있다는 주의점이 있습니다. 훈련 실행의 총 FLOPs를 계산할 수 있으면 적당한 MFU와 알려진 토폴로지를 가정하여 훈련 시간을 대략적으로 추정할 수 있습니다.</p>

### How to shard LLaMA 3-70B for training

위의 설정을 유지하여 8960개의 칩으로 구성된 TPU v5p pod에서 4M 토큰 배치 크기(배치당 길이 4096인 시퀀스 1024개)로 LLaMA 3-70B를 훈련하고 싶다고 가정해 봅시다. 이 모델에 대한 최적의 샤딩 전략이 무엇인지 논의해 봅시다.

**Question:** 위의 가정하에, FSDP만으로 모델을 훈련할 수 있을까요? 먼저 시퀀스/컨텍스트 병렬 처리를 할 수 없다고 가정해 봅시다. _이것은 간단하고 작동한다면 추가적인 통신을 도입하지 않으므로 가장 먼저 떠올려야 할 아이디어입니다._

{% details 정답을 보려면 여기를 클릭하세요 (먼저 생각해 보세요!). %}

**Answer**: 이 답변은 약간 깐깐할 수 있습니다. 위에서 언급했듯이 LLaMA 3-70B는 초기에 길이 4K의 시퀀스로 훈련되므로, 4M 토큰의 배치 크기는 1024의 *시퀀스 배치 크기*를 제공합니다. 즉, *데이터 병렬 처리를 수행해야 하는 시퀀스가 그만큼이기 때문에* 실제로는 최대 1024개의 칩까지만 순수 데이터 병렬 처리/FSDP를 수행할 수 있습니다. 따라서 "추가 통신 없는 완전 데이터 병렬 처리"라는 단순한 의미에서의 대답은 '아니요'입니다. 다음 질문은 이에 대해 약간 덜 깐깐한 버전으로 답변할 것입니다.

{% enddetails %}

**Question:** 시퀀스 샤딩을 하지 않는다는 요구 사항을 완화해 봅시다. 배치 *및* 시퀀스 축 모두에 대해 FSDP를 수행할 수 있다고 허용하면, 8960개의 칩에서 FSDP만으로 LLaMA 3-70B를 훈련할 수 있을까요?

{% details 정답을 보려면 여기를 클릭하세요 (먼저 생각해 보세요!). %}

**Answer**: 이제 시퀀스/컨텍스트 병렬 처리도 허용했으므로 훨씬 더 확장할 수 있습니다. 먼저 디바이스당 배치 크기를 계산해 봅시다. 8960방향 FSDP를 수행하면 TPU당 배치 크기는 `4 * 1024 * 1024 / 8960 = 468 토큰`이 됩니다. 이전 섹션에서 우리는 $$\text{per device batch size} < 2550 / M_X$$일 때 FSDP에 의해 ICI-bound가 된다는 것을 알았습니다. 여기서 전체 3D pod로 3개의 축을 할당할 수 있으므로, 하한은 850이 되는데, 우리는 이보다 훨씬 아래에 있습니다. **따라서 3개의 축을 사용하더라도 대답은 '아니요'입니다. 우리는 확실히 통신 병목(communication-bound) 상태가 될 것입니다.**

{% enddetails %}

**Question:** 이제 혼합 텐서 병렬 처리와 FSDP를 살펴봅시다. compute-bound를 유지할 수 있는 조합이 존재할까요? 그렇다면 FSDP와 텐서 병렬 처리를 얼마나 수행해야 할까요?

{% details 정답을 보려면 여기를 클릭하세요 (먼저 생각해 보세요!). %}

**Answer**: 먼저 이것이 맞는지 확인해 봅시다. 칩당 배치 크기가 $2550^2 / 2F = 113$보다 작으면 comms-bound가 된다는 것을 알고 있습니다. 위에서 보았듯이 우리는 이보다 약간 높습니다. 훌륭합니다! 이제 최적의 FSDP 양을 선택하기 위해 다음 공식을 사용할 수 있습니다.

$$X_{opt} = \sqrt{\frac{2BN}{F}} = \sqrt{\frac{2 \cdot 4.19e6 \cdot 8960}{28672}} = 1618$$

합리적인 2의 배수로 반올림하면, 대략 2048방향 FSDP와 4방향 텐서 병렬 처리가 나옵니다. 잘 작동할 것입니다!

{% enddetails %}

<p markdown=1 class="takeaway">**Takeaways**: 전체 TPU v5p pod에서 4M 토큰 배치 크기로 LLaMA-3를 훈련할 때, 데이터 병렬 처리(1024방향), 시퀀스 병렬 처리(2방향), 텐서 병렬 처리(4방향)를 혼합하면 통신 병목 없이 훈련할 수 있습니다. 순수 FSDP나 FSDP + 시퀀스 병렬 처리를 시도하면 comms-bound가 될 것입니다. 이전 섹션에서 우리가 만든 방정식들은 매우 실용적입니다.</p>

## Worked Problems

**Question 1 [Scaling LLaMA 70B to more chips]:** 동일한 배치 크기로 4개의 pod에서 LLaMA 3-70B를 훈련하고 싶다고 가정해 봅시다. 어떤 병렬 처리 방식을 사용해야 할까요? compute 또는 communication bound 중 어느 것이 될까요? 훈련하는 데 대략 얼마나 걸릴까요? *올바른 루프라인 한계를 사용해야 합니다.*

{% details 정답을 보려면 여기를 클릭하세요. %}

**Answer**: 4개의 pod(35,840 칩)에서 훈련하려면 DCN을 넘어야 합니다. DCN 연산 강도는 약 71,360임을 기억하세요. 4M 배치 크기를 4개의 pod에 분산하면 pod당 1M 토큰이 되며, 이는 한계를 훨씬 상회하므로 DCN 병목은 없습니다. 하지만 pod 내에서는 칩당 배치 크기가 `4M / 35840 = 111 토큰`이 됩니다. 이는 $2550^2 / 2F = 113$ 한계보다 아주 약간 낮습니다. 따라서 우리는 약간 comms-bound가 될 수 있지만, 거의 경계선에 있습니다. 최적의 전략은 가능한 한 많은 텐서 병렬 처리를 사용하는 것입니다.

훈련 시간은 칩 수가 4배 늘었으므로 대략 1/4로 줄어들어 약 11일이 걸릴 것입니다.

{% enddetails %}

**Question 2 [LLaMA 405B]:**

(a) LLaMA 3-405B [config](https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json)를 사용하여 위와 같은 주요 하이퍼파라미터 표를 작성하세요. 이 모델의 총 파라미터 수는 얼마인가요? 훈련 단계당 FLOPs는 얼마인가요? 15T 토큰에 대해 훈련하면 얼마나 많은 FLOPs를 수행하나요?

(b) 8개의 TPU v5p pod에서 훈련하고 싶다고 가정해 봅시다. 어떤 병렬 처리 방식을 사용해야 할까요? 훈련하는 데 얼마나 걸릴까요? compute 또는 comms bound 중 어느 것이 될까요?

{% details 정답을 보려면 여기를 클릭하세요. %}

(a)
| **hyperparam** | **value** |
| --------------------------- | --------- |
| $$n_\text{layers}$$ (L)     | 126       |
| $$d_\text{model}$$ (D)      | 16,384    |
| $$d_{ff}$$ (F)              | 53,248    |
| $$n_\text{heads}$$ (N)      | 128       |
| $$n_\text{kv_heads}$$ (K)   | 8         |
| $$d_\text{qkv}$$ (H)        | 128       |
| $$n_\text{embeddings}$$ (V) | 128,256   |

* FFW 파라미터: 126 * 16384 * 53248 * 3 = 3.3e11
* Attention 파라미터: 126 * (2 * 16384 * 128 * 128 + 2 * 16384 * 8 * 128) = 7.2e10
* Vocab 파라미터: 2 * 128256 * 16384 = 4.2e9
* 총 파라미터: ~406B. 거의 정확합니다!

토큰당 FLOPs: 6 * 406e9 = 2.44e12.
총 FLOPs: 2.44e12 * 15e12 = 3.66e25.

(b) 8개의 pod에는 71,680개의 칩이 있습니다. 총 훈련 시간은 `3.66e25 / (71680 * 4.59e14 * 0.4) = 2.78e6 초` 또는 약 32일입니다.
배치 크기가 4M(70B와 동일)이라고 가정하면 칩당 배치 크기는 `4M / 71680 = 55` 토큰입니다. 한계는 $2550^2 / (2 * 53248) = 61$입니다. 우리는 한계 바로 아래에 있으므로 약간 comms-bound가 될 것입니다. 하지만 16M과 같은 더 큰 배치 크기를 사용한다면(모델이 더 크기 때문에 합리적일 수 있음) 괜찮을 것입니다. 우리는 pod 간에는 데이터 병렬 처리를, pod 내에서는 혼합 FSDP/TP를 사용할 것입니다.

{% enddetails %}

<h3 markdown=1 class="next-section">섹션 6은 여기까지입니다. 트랜스포머 추론에 관한 섹션 7을 보려면 [여기](../inference)를 클릭하세요.</h3>