---
layout: distill
title: "How to Scale Your Model"
subtitle: "TPU에서의 LLM에 대한 시스템적 관점"
# permalink: /main/
description: "LLM 훈련은 종종 연금술처럼 느껴지지만 모델의 성능을 이해하고 최적화하는 것이 꼭 그럴 필요는 없습니다. 이 책은 언어 모델 스케일링의 과학을 쉽게 설명하는 것을 목표로 합니다. 즉, TPU(및 GPU)의 작동 방식과 서로 통신하는 방법, 실제 하드웨어에서 LLM이 실행되는 방식, 그리고 대규모 환경에서 효율적으로 실행되도록 훈련 및 추론 중에 모델을 병렬화하는 방법을 다룹니다. 만약 여러분이 “이 LLM을 훈련하는 데 비용이 얼마나 들까?”, “이 모델을 직접 서비스하려면 메모리가 얼마나 필요할까?”, 또는 “AllGather란 무엇일까?”와 같은 질문을 해본 적이 있다면, 이 책이 여러분에게 유용하기를 바랍니다."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

giscus_comments: true

section_number: 0

previous_section_url: ""
previous_section_name: "Part 0: Intro"

next_section_url: roofline
next_section_name: "Part 1: Rooflines"

bibliography: main.bib

citation: true

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
  - name: High-Level Outline
  - name: Links to Sections

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

{% include figure.liquid path="assets/img/dragon.png" class="img-fluid" %}

딥러닝의 많은 부분이 여전히 일종의 흑마법과 같지만, 모델 성능 최적화는, 심지어 매우 큰 규모에서도, 꼭 그럴 필요는 없습니다! 상대적으로 간단한 원칙들이 단일 가속기에서부터 수만 개에 이르기까지 모든 곳에 적용되며, 이를 이해하면 다음과 같은 많은 유용한 일들을 할 수 있습니다:

- 모델의 각 부분이 이론적 최적치에 얼마나 근접했는지 대략적으로 추정합니다.
- 다양한 규모에서 여러 병렬 처리 방식(여러 장치에 계산을 분할하는 방법)에 대해 정보에 기반한 선택을 내립니다.
- 대형 트랜스포머 모델을 훈련하고 실행하는 데 필요한 비용과 시간을 추정합니다.
- [특정](https://arxiv.org/abs/2205.14135) [하드웨어의](https://arxiv.org/abs/1911.02150) [이점을](https://arxiv.org/abs/2007.00072) 활용하는 알고리즘을 설계합니다..
- 현재 알고리즘 성능을 제한하는 요인에 대한 명확한 이해를 바탕으로 하드웨어를 설계합니다.

**사전 지식:** 독자가 LLM과 트랜스포머 아키텍처에 대한 기본적인 이해는 있지만, 대규모 환경에서의 작동 방식까지는 알지 못한다고 가정하겠습니다. LLM 훈련의 기초를 알고 있어야 하며, JAX에 대한 기본적인 지식이 있다면 이상적입니다. 유용한 배경 자료로는 트랜스포머 아키텍처에 대한 [이 블로그 게시물](https://jalammar.github.io/illustrated-transformer/)과 [트랜스포머 논문](https://arxiv.org/abs/1706.03762) 등이 있습니다. 또한 [이 목록에서](conclusion#further-reading)에서 같이 읽으면 좋을 글들을 확인해 보세요.

**목표 및 피드백:** 이 책을 다 읽고 나면, 주어진 하드웨어 플랫폼에서 트랜스포머 모델에 가장 적합한 병렬 처리 방식을 추정하고, 훈련 및 추론에 걸리는 시간을 대략적으로 계산하는 데 자신감을 느끼게 될 것입니다. 만약 그렇지 않다면, 이메일을 보내거나 댓글을 남겨주세요! 어떻게 하면 더 명확하게 만들 수 있을지 여러분의 의견을 듣고 싶습니다. 

### 왜 관심을 가져야 할까요?

3, 4년 전만 해도 대부분의 ML 연구자들은 이 책의 내용을 이해할 필요가 없었을 것이라고 생각합니다. 하지만 오늘날에는 '작은' 모델조차 하드웨어 한계에 가깝게 실행되기 때문에, 새로운 연구를 하려면 대규모 환경에서의 효율성을 고려해야 합니다.<d-footnote>역사적으로 ML 연구는 시스템 혁신과 소프트웨어 개선 사이를 번갈아 오가는 주기(tick-tock cycle)를 따라왔습니다. Alex Krizhevsky는 CNN의 속도를 높이기 위해  부정한 CUDA 코드를 작성해야 했지만, 몇 년 안에 Theano나 TensorFlow 같은 라이브러리가 등장하면서 그럴 필요가 없어졌습니다. 아마 지금의 상황에도그러한 일이 일어나 몇 년 후에는 이 책의 모든 내용이 큰 의미가 없어질지도 모릅니다. 하지만 scaling laws는 우리 모델을 끊임없이 하드웨어의 최전선으로 밀어붙였고, 가까운 미래에 최첨단 연구를 수행하는 것은 대규모 하드웨어 토폴로지에 맞춰 모델을 효율적으로 스케일링하는 방법에 대한 이해와 불가분의 관계에 놓일 가능성이 높아 보입니다.</d-footnote> **벤치마크에서 20%의 성능 향상이 루프라인(roofline) 효율성을 20% 희생시킨 대가라면 무의미합니다.** 유망한 모델 아키텍처들이 대규모 환경에서 효율적으로 실행될 수 없거나, 혹은 아무도 그렇게 만들기 위해 노력하지 않기 때문에 실패하는 경우가 허다합니다.

**'모델 스케일링'의 목표는 훈련이나 추론에 사용되는 칩의 수를 늘리면서 처리량을 비례적으로, 즉 선형적으로 증가시키는 것입니다.** 이를 "*strong scaling(강력 혹은 경성 스케일링)*"이라고 합니다. 추가 칩을 사용하는 것('병렬 처리')은 보통 계산 시간을 줄여주지만, 칩 간의 추가적인 통신 비용을 발생시킵니다. 통신이 계산보다 오래 걸리게 되면 우리는 '통신 병목(communication bound)' 상태가 되어 강력 스케일링을 할 수 없게 됩니다.<d-footnote>계산 시간이 감소함에 따라, 일반적으로 단일 칩 수준에서도 병목 현상에 직면하게 됩니다. 여러분의 최신 TPU나 GPU는 초당 500조 회의 연산을 수행하도록 평가될 수 있지만, 주의를 기울이지 않으면 메모리에서 파라미터를 옮기는 데 발목이 잡혀 그 성능의 10분의 1밖에 내지 못할 수도 있습니다. 칩당 계산 능력, 메모리 대역폭, 그리고 총 메모리 간의 상호작용은 스케일링 이야기에 매우 중요합니다.</d-footnote> 만약 우리가 하드웨어를 충분히 잘 이해하여 이러한 병목 현상이 어디서 발생할지 예측할 수 있다면, 이를 피하도록 모델을 설계하거나 재구성할 수 있습니다.<d-footnote>하드웨어 설계자들은 반대의 문제에 직면합니다. 즉, 비용을 최소화하면서 우리 알고리즘에 딱 맞는 수준의 계산 능력, 대역폭, 메모리를 제공하는 하드웨어를 구축해야 합니다. 이 '공동 설계(co-design)' 문제가 얼마나 스트레스가 심한지 상상할 수 있을 것입니다. 종종 2~3년 후에 첫 칩이 실제로 출시될 때 어떤 알고리즘이 유행할지에 대해 예측하고 투자해야 합니다. TPU의 이야기는 이 게임에서 엄청난 성공 사례입니다. 행렬 곱셈은 메모리 바이트당 사용하는 FLOPs가 다른 거의 모든 알고리즘보다 훨씬 많다는 점에서 독특한 알고리즘이며 (byte 당 N FLOPs), 초기 TPU와 그 시스톨릭 배열(systolic array) 아키텍처는 출시 당시 GPU보다 훨씬 뛰어난 가격 대비 성능(perf / $)을 달성했습니다. TPU는 ML 워크로드를 위해 설계되었고, 텐서코어(TensorCore)를 탑재한 GPU도 이 틈새 시장을 채우기 위해 빠르게 변화하고 있습니다. 하지만 만약 신경망이 성공하지 못했거나, 본질적으로 GPU보다 유연성이 떨어지는 TPU가 처리할 수 없는 근본적인 방식으로 변했다면 얼마나 큰 비용이 들었을지 상상해 볼 수 있습니다.</d-footnote>

*이 책의 목표는 TPU(및 GPU) 하드웨어의 작동 방식과, 트랜스포머 아키텍처가 현재 하드웨어에서 좋은 성능을 내기 위해 어떻게 발전해 왔는지를 설명하는 것입니다. 이 내용이 새로운 아키텍처를 설계하는 연구자들과 현 세대의 LLM을 빠르게 실행시키기 위해 노력하는 엔지니어들 모두에게 유용하기를 바랍니다.*

## High-Level Outline

이 책의 전체적인 구조는 다음과 같습니다:

[섹션 1](roofline)에서는 루프라인 분석과 스케일링을 제한할 수 있는 요인(통신, 계산, 메모리)에 대해 설명합니다. 
 [섹션 2](tpus)와 [섹션 3](sharding)에서는 TPU와 최신 GPU가 개별 칩으로서, 그리고 매우 중요하게는, 제한된 대역폭과 지연 시간을 가진 상호 연결(interconnected) 시스템으로서 어떻게 작동하는지에 대해 자세히 다루며 이를 통해 아래와 같은 질문에 답해볼 것입니다:


* 특정 크기의 행렬 곱셈은 얼마나 걸릴까요? 어느 지점부터 연산, 메모리, 또는 통신 대역폭에 의해 제한될까요?
* TPU들은 어떻게 서로 연결되어 훈련 클러스터를 형성할까요? 시스템의 각 부분은 얼마나 많은 대역폭을 가지고 있을까요?
* 여러 TPU에 걸쳐 배열(arrays)을 모으거나(gather), 흩뿌리거나(scatter), 재분배하는(re-distribute) 데 시간이 얼마나 걸릴까요?
* 여러 장치에 다르게 분산된 행렬들을 어떻게 효율적으로 곱할 수 있을까요?

{% include figure.liquid path="assets/img/pointwise-product.gif" class="img-small" caption="<b>Figure:</b> <a href=\"tpus\">섹션 2</a> 의 다이어그램으로, TPU가 elementwise product을 수행하는 방법을 보여줍니다. 배열의 크기와 다양한 링크의 대역폭에 따라, 우리는 연산 병목(하드웨어의 최대 연산 용량 사용) 상태가 되거나 통신 병목(메모리 로딩에 의해 병목 현상 발생) 상태가 될 수 있습니다." %}

5년 전 ML 분야는 ConvNet, LSTM, MLP, 트랜스포머 등 다채로운 아키텍처의 향연이었지만, 지금은 대부분 트랜스포머만 남았습니다<d-cite key="transformers"></d-cite>. 저희는 트랜스포머 아키텍처의 모든 부분을 이해하는 것이 매우 중요하다고 생각합니다. 모든 행렬의 정확한 크기, 정규화(normalization)가 일어나는 위치, 각 부분에 있는 파라미터와 FLOPs<d-footnote>FLoating point OPs, 즉 필요한 덧셈과 곱셈의 총 횟수입니다. 많은 자료에서 FLOPs를 "초당 연산 횟수"로 사용하지만, 저희는 이를 명확히 나타내기 위해 FLOPs/s를 사용합니다.</d-footnote> 의 수까지 말이죠. [섹션 4](transformers)에서는 이러한 "트랜스포머 수학"을 꼼꼼하게 다루며, 훈련과 추론 모두에 대한 파라미터와 FLOPs를 계산하는 방법을 알아봅니다. 이를 통해 우리 모델이 얼마나 많은 메모리를 사용할지, 연산이나 통신에 얼마나 많은 시간을 할애할지, 그리고 언제 attention이 feed-forward 블록에 비해 중요해질지를 알 수 있습니다.


{% include figure.liquid path="assets/img/transformer-diagram.png" class="img-fluid" caption="<b>Figure:</b> 표준 트랜스포머 레이어로, 각 행렬 곱셈(matmul)은 원 안의 점으로 표시됩니다. 모든 파라미터(정규화 제외)는 보라색으로 표시됩니다. <a href=\"transformers\">섹션 4</a> 에서 이 다이어그램을 더 자세히 살펴봅니다." %}

[섹션 5: 훈련](training) 과 [섹션 7: 추론](inference)은 이 글의 핵심으로, 다음과 같은 근본적인 질문을 다룹니다: 특정 크기의 모델과 주어진 수의 칩이 있을 때, 어떻게 모델을 병렬화하여 "strong scaling" 영역에 머무를 수 있을까? 이는 간단한 질문이지만 놀랍도록 복잡한 답을 가지고 있습니다. 넓게 보자면 모델을 여러 칩에 분산시키는 데 사용되는 4가지 주요 병렬 처리 기법(**데이터**, **텐서**, **파이프라인**, **expert**)이 있으며, 메모리 요구사항을 줄이기 위한 여러 다른 기법(**rematerialisation**, **옵티마이저/모델 샤딩(ZeRO)**, **호스트 오프로드**, **그래디언트 축적(gradient accumulation)**)이 있습니다. 여기서는 이러한 기법들 중 많은 것을 논의합니다.

위의 섹션들을 다 읽고 나면, 새로운 아키텍처나 설정에 대해 스스로 병렬 처리 방식을 선택할 수 있게 되기를 바랍니다. [섹션 6](applied-training)과 [섹션 8](applied-inference)은 이러한 개념들을 인기 있는 오픈 소스 모델인 LLaMA-3에 적용하는 실용적인 튜토리얼입니다.

마지막으로, [섹션 9](profiling) 과 [섹션 10](jax-stuff)에서는 이러한 아이디어 중 일부를 JAX에서 구현하는 방법과, 문제가 발생했을 때 코드를 프로파일링하고 디버깅하는 방법을 살펴봅니다.

책 전반에 걸쳐 스스로 풀어볼 수 있는 문제들을 제공하려고 노력했습니다. 모든 섹션을 읽거나 순서대로 읽지 않으셔도 됩니다. 또한 피드백을 부탁드립니다. 당분간 이 책은 초안이며 계속해서 수정될 예정입니다. 감사합니다!

*이 문서에 담긴 많은 아이디어를 도출해 낸 James Bradbury와 Blake Hechtman에게 감사를 표합니다.*

<h3 markdown=1 class="next-section">그럼, 서론은 이만하고, TPU 루프라인에 대한 [섹션 1로 넘어가겠습니다](roofline).</h3>

## Links to Sections

*이 시리즈는 아마 필요 이상으로 긴 것 같지만, 길이가 읽으시는데 방해물이 되지 않기를 바랍니다. 처음 세 챕터는 사전 지식이며 익숙하다면 건너뛰어도 좋지만, 이후 사용될 표기법을 소개합니다. 마지막 세 파트는 실제 모델을 다루는 방법을 설명하므로 가장 실용적일 수 있습니다.*

**Part 1: 사전 지식**

* [**챕터 1: 루프라인 분석에 대한 간략한 소개**](roofline). 알고리즘은 연산, 통신, 메모리 세 가지에 의해 제한됩니다. 이를 통하여 알고리즘이 얼마나 빨리 실행될지 근사치를 구할 수 있습니다.

* [**챕터 2: TPU를 어떻게 생각해야 할까**](tpus). TPU는 어떻게 작동할까요? TPU는 우리가 훈련하고 서빙할 수 있는 모델에 어떤 영향을 미칠까요?

* [**챕터 3: 샤딩된 행렬과 이를 곱하는 방법**](sharding). 여기서는 우리가 가장 좋아하는 연산인 (샤딩된) 행렬 곱셈을 통해 모델 샤딩과 다중 TPU 병렬 처리를 설명합니다.

**파트 2: 트랜스포머**

* [**챕터 4: 당신이 알아야 할 모든 트랜스포머 수학**](transformers). 트랜스포머는 순전파(forward pass)와 역전파(backwards pass) 과정에서 얼마나 많은 FLOPs를 사용할까요? 파라미터의 수를 계산할 수 있나요? KV 캐시의 크기는요? 여기서 이러한 수학을 다룹니다.

* [**챕터 5: 훈련을 위해 트랜스포머를 병렬화하는 방법**](training). FSDP. Megatron 샤딩. 파이프라인 병렬 처리. 주어진 수의 칩으로, 주어진 크기의 모델을 주어진 배치 크기로 가능한 한 효율적으로 훈련하려면 어떻게 해야 할까요?

* [**챕터 6: TPU에서 LLaMA 3 훈련하기**](applied-training). TPU에서 LLaMA 3를 어떻게 훈련할 수 있을까요? 시간이 얼마나 걸릴까요? 비용은 얼마나 들까요?

* [**챕터 7: 트랜스포머 추론의 모든 것**](inference). 모델을 훈련 후 서빙을 해야 합니다. 추론은 지연 시간(latency)이라는 새로운 고려 사항을 추가하고 메모리 환경을 바꿉니다. 분산 서빙(disaggregated serving)이 어떻게 작동하는지와 KV 캐시에 대해 어떻게 생각해야 하는지 이야기할 것입니다.

* [**챕터 8: TPU에서 LLaMA 3 서빙하기**](applied-inference). TPU v5e에서 LLaMA 3를 서빙하는 데 비용이 얼마나 들까요? 지연 시간과 처리량(latency/throughput)의 트레이드오프는 무엇일까요?

**파트 3: 실용적인 튜토리얼**

* [**챕터 9: TPU 코드 프로파일링 방법**](profiling). 실제 LLM은 위의 이론처럼 간단하지 않습니다. 여기서는 JAX + XLA 스택과 JAX/TensorBoard 프로파일러를 사용하여 실제 문제를 디버깅하고 수정하는 방법을 설명합니다.

* [**챕터 10: JAX로 TPU 프로그래밍하기**](jax-stuff). JAX는 계산 병렬화를 위한 마법 같은 API들을 제공하지만, 이를 사용하는 방법을 알아야 합니다. 재미있는 예제와 풀이가 있는 문제들이 있습니다.

* [**챕터 11: 결론 및 추가 자료**](conclusion). TPU와 LLM에 대한 마무리와 추가적인 읽을거리.