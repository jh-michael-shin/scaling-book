---
layout: distill
title: "How to Think About TPUs"
# permalink: /main/
description: "이 섹션에서는 TPU가 어떻게 작동하는지, 멀티칩 훈련 및 추론을 위해 어떻게 서로 연결되는지, 그리고 이가 우리가 즐겨 사용하는 알고리즘의 성능에 어떤 영향을 미치는지에 대해 자세히 다룹니다. GPU 사용자에게도 유용한 정보가 있습니다!"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting

section_number: 2

previous_section_url: "../roofline"
previous_section_name: "Part 1: Rooflines"

next_section_url: ../sharding
next_section_name: "Part 3: Sharding"

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
  - name: What Is a TPU?
  - name: TPU Networking
  - name: Key Takeaways
  - subsections:
    - name: TPU Specs
  - name: Worked Problems
  - name: Appendix
  - subsections:
    - name: "Appendix A: More on TPU internals"
    - name: "Appendix B: How does a systolic array work?"

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

## What Is a TPU?

**TPU는 기본적으로 행렬 곱셈에 특화된 연산 코어(TensorCore)가 빠른 메모리 스택(고대역폭 메모리 또는 HBM)에 부착된 형태입니다<d-cite key="tpu_paper"></d-cite>.** 아래는 TPU의 다이어그램입니다:

{% include figure.liquid path="assets/img/tpu-chip.png" class="img-fluid" caption="<b>Figure:</b> TPU 칩의 기본 구성 요소. TensorCore는 회색의 왼쪽 박스로, 행렬 곱셈 유닛(MXU), 벡터 유닛(VPU), 벡터 메모리(VMEM)를 포함합니다." %}

TensorCore는 기본적으로 정말 뛰어난 행렬 곱셈 기계라고 생각할 수 있지만, 다른 주목할 만한 몇 가지 기능도 있습니다. TensorCore에는 세 가지 핵심 유닛이 있습니다:

* **MXU** (Matrix Multiply Unit)는 TensorCore의 핵심입니다. 대부분의 TPU 세대에서, MXU는 시스톨릭 배열(systolic array)을 사용하여 8 사이클마다 `bfloat16[8,128] @ bf16[128,128] -> f32[8,128]` 행렬 곱셈<d-footnote>TPU v6e (Trillium)는 256x256 MXU를 사용하며, 이전 세대는 모두 128x128을 사용합니다.</d-footnote> 한 번 수행합니다 (<a href="#appendix-c-how-does-a-systolic-array-work">Appendix B</a> 참조).  
  * 이는 TPU v5e에서 1.5GHz로 작동할 때 MXU당 약 `5e13` bf16 FLOPs/s에 해당합니다. 대부분의 TensorCore에는 2개 또는 4개의 MXU가 있으므로, 예를 들어 TPU v5e의 총 bf16 FLOPs/s는 `2e14`입니다.  
  * TPU는 또한 더 높은 처리량을 가진 더 낮은 정밀도의 matmul도 지원합니다 (예: 각 TPU v5e 칩은 `4e14` int8 OPs/s를 수행할 수 있습니다).

* **VPU** (Vector Processing Unit)는 ReLU 활성화나 벡터 간의 원소별 덧셈 또는 곱셈과 같은 일반적인 수학 연산을 수행합니다. Reduction(sums) 연산도 여기서 수행됩니다. 자세한 내용은 <a href="#appendix-a-more-on-tpu-internals">Appendix A</a> 에서 자세히 다룹니다. 
* **VMEM** (Vector Memory)은 TensorCore 내부에 위치한 온칩(on-chip) 스크래치패드로, 연산 유닛에 근접해있습니다. HBM보다 훨씬 작지만(예: TPU v5e에서는 128MiB) MXU와의 대역폭은 훨씬 높습니다. VMEM은 CPU의 L1/L2 캐시와 꽤나 유사하게 작동하지만, 훨씬 크고 프로그래머가 제어(programmer-controlled)할 수 있습니다. HBM의 데이터는 TensorCore가 계산을 수행하기 전에 VMEM으로 복사되어야 합니다.

**TPU는 행렬 곱셈이 아주, 아주 빠릅니다**. 이는 TPU가 주로 하는 일이며, 성능 또한 뛰어납니다. 지금까지 가장 강력한 TPU 중 하나인 [TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture)는 코어당 초당 `2.5e14` bf16 FLOPs / second 또는 칩당 `5e14` bf16 FLOPs / second 을 수행할 수 있습니다. 8960개 칩으로 구성된 단일 pod는 초당 4 exaflops를 처리할 수 있습니다. 이는 *어마어마한* 양입니다. 이는 세계에서 가장 강력한 슈퍼컴퓨터 중 하나이며, 구글은 이를 다수 보유하고 있습니다.<d-footnote>TPU와 특히 이의 시스톨릭 배열(systolic arrays)이 이토록 강력한 하드웨어 가속기인 이유는, 행렬 곱셈이 $O(n^2)$ 바이트에 대해 $O(n^3)$의 연산을 사용하는 몇 안 되는 알고리즘 중 하나이기 때문입니다. 이로 인해 일반적인 ALU가 메모리 대역폭이 아닌 연산 자체에 의해 병목 현상을 겪기 매우 쉽습니다.</d-footnote>

위의 다이어그램에는 제어 흐름 처리(control flow handling)에 사용되는 SMEM 및 스칼라(scalar) 유닛과 같은 몇 가지 다른 구성 요소도 포함되어 있으며, 이는 <a href="#appendix-a-more-on-tpu-internals">Appendix A</a>에서 짧게 다루지만, 이해하는 데 필수적이지는 않습니다. 반면에 HBM은 중요하면서 또한 비교적 간단합니다:

* **HBM** (High Bandwidth Memory) 은 TensorCore에서 사용할 텐서를 저장하는 큰 용량의 빠른 메모리입니다. HBM은 보통 수십 기가바이트의 용량을 가집니다(예를 들자면, [TPU v5e는 16GiB의 HBM을 가짐](https://cloud.google.com/tpu/docs/v5e#system_architecture)).

  * 계산이 필요할 때, 텐서는 HBM에서 VMEM(아래 예제 있음)을 통해 스트리밍되어 MXU로 들어가고, 결과는 VMEM에서 다시 HBM으로 쓰입니다.
  
  * HBM과 TensorCore(VMEM을 통해) 간의 대역폭은 "HBM 대역폭” (보통 1-2TB/sec)이라 하며, 메모리 병목(memory-bound) 워크로드에서 계산이 얼마나 빨리 수행할 수 있는지의 제약 사항이 됩니다. 

**보통 모든 TPU 연산은 파이프라인화되고 중첩됩니다.** matmul $X \cdot A \to Y$ 를 수행하기 위해, TPU는 먼저 HBM에서 $A$ 와 $X$ 행렬의 청크를 VMEM으로 복사한 다음, 이를 MXU로 로드하여 8x128($X$의 경우) 및 128x128($A$의 경우) 청크를 곱하고, 그 결과를 청크 단위로 다시 HBM에 복사합니다. 이를 효율적으로 수행하기 위해, matmul은 VMEM으로/에서 복사하는 작업이 MXU 작업과 중첩되도록 파이프라인화됩니다. 이를 통해 MXU는 메모리 전송을 기다리지 않고 계속 작동할 수 있으며, matmul이 메모리 병목이 아닌 연산 병목 상태를 유지하게 합니다.

다음은 HBM에서 원소별 곱셈(elementwise product)을 수행하는 방법의 예제입니다:

{% include figure.liquid path="assets/img/pointwise-product.gif" caption="<b>Figure:</b> HBM에서 바이트를 로드하여 TPU에서 원소별 곱셈(pointwise product)을 수행하는 애니메이션. 바이트가 메모리에서 청크 단위로 스트리밍되고, 전체 배열이 구체화되기를 기다리지 않고 부분 결과가 파이프라인으로 다시 전송되는 방식을 주의 깊게 봐주세요." %}

matmul은 VPU/Vector Unit 대신 MXU로 로드되고, 동일한 가중치 청크가 여러 활성화 청크에 사용되므로 로드 및 저장 순서가 다르다는 점을 제외하면 거의 동일하게 보일 것입니다. 데이터 청크가 VMEM으로, 다음 VREG(vector registers)로, 다음 Vector Unit으로, 그리고 다시 VMEM과 HBM으로 스트리밍되는 것을 볼 수 있습니다. 곧 보게 되겠지만, HBM에서 VMEM으로의 로드가 Vector Unit(또는 MXU)의 FLOPs보다 느리면, VPU나 MXU에 작업이 공급되지 않아 "대역폭 병목" 상태가 됩니다.

<p markdown=1 class="takeaway">**Key takeaway:** TPU는 아주 심플합니다. HBM에서 VMEM으로 가중치를 로드한 다음, VMEM에서 초당 약 200조 번의 multiply-adds 연산을 수행할 수 있는 시스톨릭 배열로 로드합니다. HBM $\leftrightarrow$ VMEM 그리고 VMEM $\leftrightarrow$ 시스톨릭 배열 대역폭은 TPU가 효율적으로 수행할 수 있는 계산에 대한 근본적인 한계를 설정합니다.</p>

**VMEM과 arithmetic intensity:** VMEM은 HBM보다 훨씬 작지만 MXU로의 대역폭은 훨씬 높습니다. [섹션 1](../roofline)에서 보았듯이, 이는 알고리즘의 모든 입력/출력을 VMEM에 맞출 수 있다면 통신 병목에 부딪힐 가능성이 훨씬 작아진다는 것을 의미합니다. 이는 계산의 arithmetic intensity가 낮을 때 특히 유용합니다: 

VMEM 대역폭은 HBM 대역폭보다 약 22배 높으므로, VMEM에서 읽고 쓰는 MXU 연산은 최대 FLOPs 활용도를 달성하기 위해 10-20의 arithmetic intensity만 필요합니다. 즉, 가중치를 HBM 대신 VMEM에 맞출 수 있다면, 훨씬 작은 배치 크기에서도 행렬 곱셈이 FLOPs 병목 상태가 될 수 있습니다. 그리고 근본적으로 낮은 arithmetic intensity를 가진 알고리즘도 여전히 효율적일 수 있다는 의미입니다. 다만 VMEM이 너무 작아서 이것이 종종 어려운 과제가 됩니다.<d-footnote>우리는 때때로 VMEM 프리페칭(prefetching)에 대해 이야기하는데, 이는 matmul에 대한 로딩 비용을 가리기 위해 VMEM에 가중치를 미리 로드하는 것을 의미합니다. 예를 들어, 일반적인 Transformer에서 어텐션 중에 큰 피드포워드 가중치를 VMEM으로 로드하여, 메모리 대역폭 병목 상태일 경우 가중치 로드 비용을 숨길 수 있습니다. 이를 위해서는 가중치가 충분히 작거나, 단일 레이어를 VMEM에 넣고도 공간이 남을 만큼 충분히 샤딩되어야 합니다.</d-footnote>

{% include figure.liquid path="assets/img/tpu-bandwidth.png" class="img-fluid" %}

**TPU 칩은 일반적으로(항상 그런 것은 아니지만) 메모리를 공유하는 두 개의 TPU 코어로 구성되며, 두 배의 FLOPs를 가진 하나의 큰 가속기("메가코어(megacore)" 구성)로 간주될 수 있습니다.** TPU v4 이후로는 이렇게 구성되어 있습니다. 구형 TPU 칩(TPU v3 및 이전)은 메모리가 분리되어 있으며 두 개의 별도 가속기로 간주됩니다. TPU v5e와 같은 추론에 최적화된 칩은 칩당 하나의 TPU 코어만 가지고 있습니다.

{% include figure.liquid path="assets/img/cores.png" class="img-fluid img-small" %}

**칩**은 **'트레이(tray)' 위에 4개 세트**로 배열되어 **PCIe 네트워크를 통해 CPU 호스트에 연결**됩니다. Colab이나 단일 TPU-VM을 통해 4개의 칩(8개 코어지만, 보통 4개의 논리적 메가코어로 취급됨)이 노출되는 이 형식이 대부분의 독자에게 익숙할 것입니다. TPU v5e와 같은 추론 칩의 경우, 호스트당 1개가 아닌 2개의 트레이가 있지만, 칩당 코어는 1개뿐이므로 8개 칩 = 8개 코어가 됩니다.<d-footnote>Cloud TPU VM에서는 각 트레이가 별도의 VM의 일부로 노출되므로, 다시 4개의 코어만 보입니다.</d-footnote>

{% include figure.liquid path="assets/img/pcie.png" class="img-fluid" %}

**PCIe 대역폭은 제한적입니다:** HBM $\leftrightarrow$ VMEM 링크와 마찬가지로, CPU $\leftrightarrow$ HBM PCIe 연결은 호스트 메모리에서 HBM으로 또는 그 반대로 얼마나 빨리 로드할 수 있는지를 제한하는 특정 대역폭을 가집니다. 예를 들어, TPU v4의 PCIe 대역폭은 각 방향으로 초당 16GB이므로, HBM보다 거의 100배 느립니다. 우리는 호스트(CPU) RAM으로 데이터를 로드/오프로드할 수 *있지만*, 그다지 빠르지는 않습니다.

## TPU Networking

**칩은 Pod 내에서 ICI 네트워크를 통해 서로 연결됩니다.** 구형 세대(TPU v2 및 TPU v3), 추론 칩(예: TPU v5e), 그리고 Trilium (TPU v6e)에서, ICI("inter-chip interconnects")는 가장 가까운 4개의 이웃을 연결합니다(edge 링크로 2D torus를 형성함). TPU v4와 TPU v5p는 가장 가까운 6개의 이웃에 연결됩니다(3D torus를 형성함). 이러한 연결은 호스트를 통하지 **않고**, 칩 간의 직접적인 링크라는 점에 유의하세요.

{% include figure.liquid path="assets/img/ici-wraparound.png" class="img-fluid img-small" %}

토로이드(toroidal) 구조는 임의의 두 노드 간의 최대 거리를 $N$ 에서 $N / 2$ 로 줄여 통신을 훨씬 빠르게 만듭니다. TPU는 또한 노드 간의 평균 거리를 더욱 줄이기 위해 뫼비우스의 띠와 같은 토폴로지로 토러스를 감싸는 "트위스티드 토러스(twisted torus)" 구성을 가지고 있습니다.

**TPU pod(ICI로 연결된)는 엄청 커질 수 있습니다:** 최대 pod 크기(**superpod**이라고 함)는 TPU v4의 경우 `16x16x16`이고 TPU v5p의 경우 `16x20x28`입니다. 이러한 대규모 pod는 매우 큰 토폴로지를 연결하기 위해 재구성할 수 있는 [optical wraparound links](https://arxiv.org/pdf/2208.10041)<d-footnote>광학 스위치는 동일한 ICI 대역폭을 가진 재구성 가능한 연결일 뿐입니다. 이를 통해 랩어라운드 링크를 유지하면서 큐브를 연결할 수 있습니다.</d-footnote>로 연결된 `4x4x4` 칩의 재구성 가능한 큐브로 구성됩니다.

{% include figure.liquid path="assets/img/tpu-rack.png" class="img-fluid" %}

더 작은 토폴로지(예: `2x2x1`, `2x2x2`)도 요청할 수 있지만, 랩어라운드(wraparound)는 제공되지 않습니다. 이는 대부분의 통신 시간을 일반적으로 두 배로 만들기 때문에 중요한 주의 사항입니다. 풀 큐브(full cube)의 배수(예: `4x4x4` 또는 `4x4x8`)는 광학 스위치에 의해 제공되는 랩어라운드를 가집니다.<d-footnote>`2x2x4` 토폴로지에는 랩어라운드가 없습니다. 랩어라운드는 풀 큐브에서만 사용 가능한 광학 스위치로 제공되기 때문입니다. 하지만 TPU v5e 8x16은 재구성 가능한 광학 네트워킹(reconfigurable optical networking)을 사용하지 않으므로, 긴 축에는 랩어라운드가 있습니다.</d-footnote>

{% include figure.liquid path="assets/img/subslices.png" class="img-fluid" %}

TPU v5e와 Trillium pod는 크기가 16인 축을 따라 랩어라운드가 있는 단일 `16x16` 2D 토러스로 구성됩니다(즉, `8x16`은 긴 축에 랩어라운드가 있음). TPU v5e와 v6e(Trillium)는 16x16 토러스를 넘어 확장할 수 없지만, pod들은 TPU 호스트를 서로 연결하는 표준 데이터센터 네트워킹(DCN)을 통해 여전히 서로 통신할 수 있습니다. 다시 말하지만, 16 미만($<16$)의 차원에는 랩어라운드 없이 더 작은 토폴로지를 요청할 수 있습니다.

{% include figure.liquid path="assets/img/more-subslices.png" class="img-fluid" %}

**최근접 이웃(nearest-neighbor) 연결성은 TPU와 GPU의 핵심적인 차이점입니다.** TPU처럼 로컬 연결을 사용하는 대신, GPU는 모든 GPU 간의 점대점(point-to-point) 연결을 근사화하는 계층적 스위치로 연결됩니다. 일반적으로 노드 내의 GPU(H100의 경우 8개, B200의 경우 최대 500개)는 직접 연결되지만, 더 큰 토폴로지에서는 각 GPU 간에 O(log(N)) 홉이 필요합니다. 한편으로는, 이는 GPU가 노드 내에서 임의의 데이터를 단일 저지연 홉(low-latency hop)으로 보낼 수 있음을 의미합니다. 다른 한편으로는, TPU가 훨씬 저렴하고(NVLink 스위치는 비쌉니다) 함께 연결하기가 더 간단하며, 장치당 링크 수와 장치당 대역폭이 일정하기 때문에 훨씬 더 큰 토폴로지로 확장할 수 있습니다.

**ICI는 DCN에 비해 매우 빠르지만, HBM 대역폭보다는 여전히 느립니다.** 예를 들어, [TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture)는 :

* 칩당 `2.5e12` bytes/s (2.5 TB/s)의 HBM 대역폭.
* 축당(칩당 3개 축) `9e10` bytes/s (90<d-footnote>위의 페이지에는 100 GB/s의 대역폭이라고 적혀있어, 글에서 명시된 것과 조금 다릅니다. TPU ICI 링크는 수행되는 연산에 따라 대역폭이 약간 다르기에 전반적으로 이 문서의 수치를 우려 없이 사용하셔도 됩니다.</d-footnote> GB/s)의 ICI 대역폭.
* 호스트당 `2.5e10` bytes/s (25 GB/s) of DCN (egress) 대역폭. 일반적으로 호스트당 8개의 TPU가 있으므로, 이는 실제로는 `3.1e9` bytes / s / chip에 가깝습니다.

이는 모델을 여러 칩에 분산시킬 때, 더 느린 장치 간 통신으로 MXU에 병목이 생기지 않도록 주의해야 함을 의미합니다.

**Multi-slice training:** ICI로 연결된 TPU 세트를 **slice** 라고 합니다. 각각의 슬라이스들은 DCN을 사용하여 서로 연결될 수 있는데, 예를 들어 다른 pod에 있는 슬라이스들을 연결하는 경우입니다. DCN은 ICI보다 훨씬 느린 연결이므로, 계산이 DCN 데이터 때문에 지연되는 시간을 최소화해야 합니다. DCN은 호스트 대 호스트(host-to-host)이므로, DCN을 통해 TPU에서 TPU로 버퍼를 전송하려면 먼저 PCIe를 통해 호스트로 전송한 다음, 네트워크를 통해 송신하고, 대상 호스트 네트워크를 통해 수신한 다음, PCIe를 통해 HBM으로 전송해야 합니다.

## Key Takeaways

* TPU는 간단하며, 대부분의 경우 메모리(매우 빠름), ICI를 통해 다른 칩(상당히 빠름), DCN을 통해 데이터센터의 나머지 부분(어느 정도 빠름)에 연결된 행렬 곱셈 유닛으로 생각할 수 있습니다.

* 통신은 속도 순서대로 다양한 네트워크 대역폭에 의해 제한됩니다:
  * HBM 대역폭: TensorCore와 관련 HBM 사이.
  * ICI 대역폭: TPU 칩과 가장 가까운 4개 또는 6개의 이웃 사이.
  * PCIe 대역폭: CPU 호스트와 관련 칩 트레이(들) 사이.
  * DCN 대역폭: 여러 CPU 호스트 사이, 일반적으로 ICI로 연결되지 않은 호스트.

* **슬라이스 내에서, TPU는 ICI를 통해 가장 가까운 이웃(nearest neighbors)에만 연결됩니다.** 이는 슬라이스 내의 멀리 떨어진 칩 간의 ICI 통신은 먼저 중간(intervening) 칩을 거쳐야 함을 의미합니다.

* **가중치 행렬은 MXU를 완전히 채우기 위해 양쪽 차원 모두 최소 128(TPU v6의 경우 256)의 크기로 패딩되어야 합니다** (실제로 더 작은 축은 128로 패딩됩니다).

* **낮은 정밀도의 행렬 곱셈이 더 빠른 경향이 있습니다.** TPU는 이를 지원하는 세대에서 bfloat16 FLOPs보다 약 2배/4배 빠른 int8 또는 int4 FLOPs를 수행할 수 있습니다. VPU 연산은 여전히 fp32에서 수행됩니다.

* TPU 연산 유닛에 병목이 생기는 것을 피하기 위해, **각 채널을 통한 통신량이 그 속도에 비례하도록 해야 합니다**.

* **다음은 우리 칩에 대한 몇 가지 구체적인 수치입니다:**

| Model                                      | Pod size | Host size | HBM capacity/chip | HBM BW/chip (bytes/s) | FLOPs/s/chip (bf16) | FLOPs/s/chip (int8) |
| :----------------------------------------- | :------: | :-------: | :---------------: | :-------------------: | :-----------------: | :-----------------: |
| <span class="nowrap-header">TPU v3</span>  |  32x32   |    4x2    |       32GB        |        9.0e11         |       1.4e14        |       1.4e14        |
| <span class="nowrap-header">TPU v4p</span> | 16x16x16 |   2x2x1   |       32GB        |        1.2e12         |       2.75e14       |       2.75e14       |
| <span class="nowrap-header">TPU v5p</span> | 16x20x28 |   2x2x1   |       96GB        |        2.8e12         |       4.59e14       |       9.18e14       |
| <span class="nowrap-header">TPU v5e</span> |  16x16   |    4x2    |       16GB        |        8.1e11         |       1.97e14       |       3.94e14       |
| <span class="nowrap-header">TPU v6e</span> |  16x16   |    4x2    |       32GB        |        1.6e12         |       9.20e14       |       1.84e15       |

호스트 크기는 단일 호스트에 연결된 TPU의 토폴로지를 나타냅니다(예: TPU v5e는 4x2 토폴로지의 8개 TPU에 연결된 단일 CPU 호스트를 가짐). 다음은 interconnect 수치입니다:

| Model       | ICI BW/link (one-way, bytes/s) | ICI BW/link (bidi, bytes/s) |
| :---------- | :----------------------------: | :-------------------------: |
| **TPU v3**  |              1e11              |            2e11             |
| **TPU v4p** |             4.5e10             |            9e10             |
| **TPU v5p** |              9e10              |           1.8e11            |
| **TPU v5e** |             4.5e10             |            9e10             |
| **TPU v6e** |              9e10              |           1.8e11            |

단방향(unidirectional) 대역폭이 하드웨어에 더 충실하지만, 완전한 링(full ring)을 포함하는 상황에서는 양방향(bidirectional) 대역폭이 더 자주 등장하므로 두 가지를 모두 포함합니다.<d-footnote>양방향(bidirectional) 대역폭이란 단일 링크를 따라 양방향으로 전송될 수 있는 총 바이트 수, 또는 동등하게, 두 링크를 효율적으로 사용할 수 있다고 가정할 때 특정 축을 따라 단일 TPU에서 나가는 총 바이트 수를 의미합니다. 이는 작동하는 링(functioning ring), 즉 특정 축에 랩어라운드(wraparound) 연결이 있을 때 해당됩니다. 이는 추론 칩에서 전체 16개 축이 있을 때, 또는 훈련 칩(v*p)에서 축이 4의 배수일 때 발생합니다. 양방향 통신을 포함하는 계산에 자주 등장하기 때문에 양방향 대역폭을 사용하는 것을 선호합니다.</d-footnote>

PCIe 대역폭은 일반적으로 칩당 `1.5e10` bytes / second 정도이며<d-footnote>Trillium (TPU v6e)는 32GB/s로, v5보다 약 2배 높습니다.</d-footnote>, DCN 대역폭은 일반적으로 호스트당 `2.5e10` bytes / second 정도입니다. 완전성을 위해 단방향 및 양방향 대역폭을 모두 포함합니다. 일반적으로 완전한 랩어라운드 링에 접근할 수 있을 때 양방향 대역폭이 더 유용한 수치이며, 단방향 대역폭은 하드웨어에 더 충실합니다.

## Worked Problems

이 숫자들은 약간 지루하지만, 모델 성능에 대한 기본적인 루프라인 추정을 가능하게 합니다. 이것이 왜 유용한지 설명하기 위해 몇 가지 문제를 풀어보겠습니다. 파트 3에서 더 많은 예시를 볼 수 있습니다.

**문제 1 [bounding LLM latency]:** 32개의 TPU v4p에 분산된 bf16의 200B 파라미터 모델에서 샘플링하고 싶다고 가정해 봅시다. HBM에서 시스톨릭 배열로 모든 파라미터를 로드하는 데 얼마나 걸릴까요? *힌트: 위의 수치를 사용하세요.*

{% details 답을 보려면 여기를 클릭하세요. %}

**답:** 32개 칩에 `sizeof(bf16) * 200e9 = 400e9` 바이트를 로드하고 있으므로, 칩당 12.5e9 바이트이며, 각 칩의 HBM 대역폭은 1.23e12입니다. 따라서 로드에는 약 10ms가 걸립니다.

꽤 멋지죠, 왜냐하면 *이것이 모델에서 샘플링하는 지연 시간의 합리적인 하한*이기 때문입니다. 각 샘플링 단계는 HBM에서 모든 파라미터를 로드해야 하므로 10ms보다 적게 걸릴 수 없습니다. 실제로, 작은 배치 크기에서는 이 값에 가깝게 달성할 수 있습니다.

{% enddetails %}

**문제 2 [TPU details]:** 완전한 TPU v5e pod를 고려해 봅시다. 총 CPU 호스트는 몇 개입니까? TPU TensorCore는 몇 개입니까? 전체 pod의 총 FLOPs/s는 얼마입니까? 총 HBM은 얼마입니까? TPU v5p pod에 대해서도 동일한 연습을 해보세요.

{% details 답을 보려면 여기를 클릭하세요. %}

**답:** TPU v5e의 경우, 각 pod는 `16x16`이고 각 호스트는 4x2 슬라이스이므로, `16*16 / 8 = 32`개의 호스트가 있습니다. TPU v5e의 경우, 각 TPU는 하나의 코어만 가지고 있으므로, 256개의 TensorCore가 있습니다. 총 FLOPs/s는 bfloat16에서 `16*16*2e14 = 5.1e16`입니다. 각 칩은 16GB의 HBM을 가지고 있으므로, 총 메모리는 `256 * 16 = 4TB`입니다.

완전한 TPU v5p pod의 경우, `16x20x28`개의 칩이 있고 각 호스트는 2x2x1이므로, `16*20*28 / 2*2 = 2,240`개의 호스트가 있습니다. TPU v5p의 경우, 각 TPU는 두 개의 TensorCore를 가지고 있으므로, `8960 * 2 = 17,920`개의 코어가 있습니다. 총 FLOPs/s는 bfloat16에서 `8960 * 4.5e14 = 4e18`입니다. 각 칩은 96GB의 HBM을 가지고 있으므로, 총 메모리는 `8960 * 96 = 860TB`입니다.

{% enddetails %}

**문제 3 [PCIe operational intensity]:** $\text{bfloat16}[D, F]$ 타입의 큰 가중치 행렬 $A$와 $\text{bfloat16}[B, D]$ 타입의 활성화 배치 $x$를 호스트 DRAM에 저장하고, 이에 대한 행렬 곱셈을 수행해야 한다고 상상해 봅시다. 이는 단일 호스트에서 실행되며, 여기에 연결된 단일 TPU v6e 칩을 사용합니다. $B \\ll D$이고 $F = 4D$라고 가정할 수 있습니다(이러한 가정이 왜 합리적인지는 향후 챕터에서 확인해볼 수 있습니다). PCIe를 통해 FLOPs 병목 상태를 유지하기 위해 필요한 가장 작은 배치 크기 $B$는 얼마입니까? PCIe 대역폭을 초당 1.5e10 바이트로 가정합니다.

{% details 답을 보려면 여기를 클릭하세요. %}

**답:** $2BDF$개의 부동소수점 연산을 수행해야 하며, 각 칩은 초당 `9.2e14`개의 부동소수점 연산을 수행할 수 있습니다. 따라서 이를 수행하는 데 $2BDF / 9.2e14$초가 걸립니다. DRAM에서 $2DF + 2BD$ 바이트를 로드하고, $2BF$ 바이트를 다시 써야 합니다. PCIe 전송 속도에 의해 병목이 발생하므로, TPU로/에서 데이터를 전송하는 데 $2 \cdot (BD + DF + BF) / 1.5e10$초가 필요합니다. 
모든 가중치 로딩을 계산과 중첩시킬 수 있다고 가정할 때, 계산이 가중치 로딩보다 오래 걸리게 하려면 다음 부등식이 성립해야 합니다: $2BDF / 9.2e14 > 2 \cdot (BD + DF + BF) / 1.5e10$ . $B \ll D$이고 $F = 4D$라는 가정을 사용하여 이를 단순화하면 다음과 같습니다.

$$\frac{8BD^2}{9.2 \times 10^{14}} > \frac{8D^2}{1.5 \times 10^{10}}$$

또는

$$B > \frac{9.2 \times 10^{14}}{1.5 \times 10^{10}} \simeq 61{,}000$$

{% enddetails %}

**문제 4 [general matmul latency]:** 가중치 행렬 int8[16384, 4096]을 알 수 없는 배치 크기 B를 가진 활성화 행렬 int8[B, 4096]과 곱하고 싶다고 가정해 봅시다. 먼저 1개의 TPUv5e에서 실행한다고 가정합니다.

1. 이 곱셈은 B의 함수로서 얼마나 오래 걸릴까요? *Hint: 배열을 HBM에서 로드하는 데 걸리는 시간과 곱셈이 실제로 걸리는 시간을 계산하는 것이 도움이 될 수 있습니다. 어느 쪽이 병목입니까?* 
2. 이 연산을 VMEM에서 실행하고 싶다면 어떨까요? B의 함수로서 얼마나 오래 걸릴까요?

{% details 답을 보려면 여기를 클릭하세요. %}

**답:** (1) 수행해야 하는 부동소수점 연산의 수는 $2 \cdot 4096 \cdot 16384 \cdot B = 1.3e8 \cdot B$입니다. 따라서 $T_{\text{math}} = (1.3e8 \cdot B) / 3.94e14$초입니다. HBM에서 VMEM으로 $16384 \cdot 4096 + 4096 \cdot B$ 바이트를 로드하고, VMEM에서 HBM으로 $16384 \cdot B$ 바이트를 다시 써야 합니다. 이는 $T_{\text{comms}} = (6.7e7 + 2e4\cdot B) / 8.1e11$초를 의미합니다. 통신과 계산이 최대한 중첩된다고 가정하면, 전체 곱셈은 대략 다음과 같이 걸릴 것입니다.

$$\max\{T_{\text{math}}, T_{\text{comms}}\} = \max\left\{ \frac{6.7 \times 10^{7} + 2 \times 10^{4} \cdot B}{8.1 \times 10^{11}}, \frac{1.3 \times 10^{8} \cdot B}{3.94 \times 10^{14}} \right\}$$

$\frac{6.7e7 + 2e4\cdot B}{8.1e11} < \frac{1.3e8 \cdot B}{3.94e14}$일 때, 즉 $B > 271$일 때 FLOPs 병목 상태가 됩니다. 이는 $$D$$와 $$F$$의 전체 영향을 고려했기 때문에 아래에서 유도하는 240이라는 숫자보다 약간 큽니다.

(2) 대신 VMEM에서 로드하는 경우, MXU에 대한 VMEM 대역폭을 HBM $\leftrightarrow$ VMEM 대역폭의 22배로 간주합시다. 이렇게 하면 데이터 로딩 분모가 8.1e11에서 1.78e13으로 바뀌고, $B > 11$을 얻습니다. 실제로, 모든 VMEM 대역폭을 $W$ 로드에 할당할 수 없으므로 실제로는 20에 가까울 것입니다.

{% enddetails %}

**문제 5 [ICI bandwidth]:** TPU v5e `4x4` 슬라이스가 있다고 가정해 봅시다. `TPU{0,0}`에서 `TPU{3, 3}`으로 `bfloat16[8, 128, 8192]` 타입의 배열을 보내고 싶다고 가정해 봅시다. TPU v5e의 홉당 지연 시간은 $1\mu s$라고 가정합니다.

1.  첫 번째 바이트가 목적지에 도착하는 데 얼마나 걸릴까요?
2.  전체 전송은 얼마나 걸릴까요?

{% details 답을 보려면 여기를 클릭하세요. %}

**답:** TPUv5e에는 2D 연결성이 있습니다. `4x4` 슬라이스(크기가 16인 축 없음)만 있기 때문에 랩어라운드 연결이 없습니다. 따라서 대상 칩이 데이터를 수신할 수 있는 포트는 두 개이고, 마찬가지로 소스 칩이 데이터를 보낼 수 있는 포트도 두 개입니다. 전송해야 하는 데이터 양은 `2 * 8 * 128 * 8192 = 1.7e7` 바이트입니다. 두 포트에서 동시에 전송할 수 있으므로(즉, 배열의 절반을 오른쪽으로, 절반을 아래로 보냄), 초당 `2 * 4.5e10 = 9e10` 바이트가 전송되며, 이는 전체 배열을 전송하는 데 약 `1.7e7 / 9e10 = 188us`가 걸릴 것임을 의미합니다(대역폭 병목이라고 가정). `4x4` 슬라이스에서는 16개 미만의 칩을 가진 축에 대한 랩어라운드 링크가 없으므로 칩 $(0, 0)$과 $(3, 3)$ 사이에는 6개의 홉이 있습니다. 각 홉의 지연 시간은 약 $1\mu s$이므로, 첫 번째 바이트는 약 `6us` 후에 도착하고 전체 전송에는 `188us`가 걸릴 것입니다.

{% enddetails %}

**문제 6 [pulling it all together, hard]:** TPU v5e 4x4 슬라이스에 걸쳐 균등하게 샤딩되었지만 각 칩의 호스트 DRAM에 오프로드된 큰 행렬 **A**: `int8[128 * 1024, 128 * 1024]`가 있다고 상상해 봅시다. 전체 배열을 TPU{0, 0}으로 복사하고 벡터 `bf16[8, 128 * 1024]`와 곱하고 싶다고 가정해 봅시다. 얼마나 오래 걸릴까요? *힌트: 위의 수치를 사용하세요.*

{% details 답을 보려면 여기를 클릭하세요. %}

**답:** 수행해야 할 작업을 개괄하는 것으로 시작하겠습니다. 우리 배열은 약 16GB입니다. 위 표에서, TPU v5e 호스트는 4x2 토폴로지를 가지고 있으므로, 4x4는 2개의 호스트를 가집니다. 따라서, 배열이 균등하게 샤딩되었으므로 각 호스트는 효과적으로 배열의 1/2, 즉 8GB의 청크를 포함합니다. 이 청크들을 모두 TPU{0,0}으로 복사해야 하며, 여기에는 두 가지 옵션이 있습니다:

1.  DCN을 통해 복사한 다음 전체 샤딩되지 않은 배열을 PCIe를 통해 HBM으로 로드할 수 있습니다.
2.  샤딩된 배열을 해당 TPU에 로드한 다음, ICI를 통해 gather를 수행하고, TPU{0,0}에서 matmul을 수행할 수 있습니다.

옵션 (2)가 더 낫다는 것이 분명합니다. DCN은 ICI에 비해 느리고, 몇 개의 PCIe 링크(호스트 0의 8개)가 아닌 많은 PCIe 링크를 통해 큰 배열을 로드하는 것을 훨씬 선호합니다. 다음은 시스템 일부의 다이어그램입니다. 위에서 설명한 대로, TPU는 ICI를 통해 이웃에 연결되고(호스트 간에도), 모든 TPU는 호스트 CPU에 연결되며(PCIe를 통해), 호스트는 DCN으로 연결됩니다.

{% include figure.liquid path="assets/img/challenge-problem.png" class="img-fluid img-small" caption="명확성을 위해 여기에는 하나만 표시되었지만, 실제로는 각 칩이 호스트에 자체 PCIe 링크를 가지고 있습니다." %}

이제 각 부분이 얼마나 걸릴지 살펴보겠습니다:

1.  **PCIe 로드**: 16개의 PCIe 링크를 통해 16GB / 2 = 8GB의 청크를 로드하고 있으며, 각 링크는 초당 `1.5e10` 바이트의 대역폭을 가집니다. 따라서 이는 약 33ms가 걸릴 것입니다.

2.  **ICI 복사:** 각 TPU는 이제 배열의 16GB / 16 = 1GB를 가지고 있습니다. ICI 대역폭은 링크당 *양방향*으로 초당 9e10 바이트이며, 위 다이어그램에서 TPU{0,0}의 경우 이 토폴로지에서 4개의 ICI 링크 중 2개만 사용 중임을 알 수 있습니다. TPU{0,0}이 2개의 축을 따라 링크당 `4.5e10` bytes/s로 총 15GB를 수신해야 하므로, 시간의 하한은 `15e9 / (4.5e10 * 2) = 167ms`로 정할 수 있습니다. 실제로 로드가 매우 고르지 않기 때문에 이는 아마도 달성할 수 없겠지만, 아마도 2배 이내일 것입니다. 섹션 2에서 보게 되겠지만, 전체 AllGather를 수행하는 데도 대략 `16e9 / (4.5e10 * 2)`가 걸릴 것이므로, 이것은 최적에 가깝습니다.

3. **HBM $\rightarrow$ MXU load:** 최종 matmul을 수행하려면, 이 16e9 바이트와 bf16[8, 128 \* 1024] 배열(또 다른 2MB이므로 무시 가능)을 HBM 대역폭을 통해 MXU로 로드해야 하며, 이는 `16e9 / 8.1e11 = 19ms`가 걸릴 것입니다.

4.  **FLOPs:** 총 $$2 \cdot 8 \cdot 128 \cdot 1024 \cdot 128 \cdot 1024 = 2.7e11$$ FLOPs를 수행하고 있으며, 초당 `1.97e14` bf16 FLOPs를 수행할 수 있으므로 1.3ms를 얻습니다.

총 시간에 대한 상한은 이 모든 시간의 합이지만, TPU가 일반적으로 이러한 작업을 중첩시킬 수 있으므로, 이는 가장 느린 부분이 병목이 되는 파이프라이닝 문제로 볼 수 있습니다. 그것이 사실이라고 가정하면, 답은 약 150-200ms입니다.

{% enddetails %}

<h3 markdown=1 class="next-section">That's it for Part 2! For Part 3, covering partitioning and cross-TPU communication, [click here](../sharding).</h3>

<h3 markdown=1 class="next-section">파트 2는 여기까지입니다! 파티셔닝과 교차 TPU 통신을 다루는 파트 3을 보려면, [여기를 클릭하세요](../sharding).</h3>

## Appendix

### Appendix A: More on TPU internals

여기서는 TPU의 내부 작동에 대해 더 깊이 파고들어 보겠습니다. 별도의 언급이 없는 한, TPU v5p의 사양을 기준으로 설명합니다.

### VPU

VPU는 TPU의 벡터 연산 코어(vector arithmetic core)입니다. VPU는 vadd(vector addition)나 vmax(elementwise max) 같은 원소별 산술 연산을 수행하는 2차원 SIMD 벡터 머신(**VPU**)과, VPU 및 MXU를 위한 데이터를 담는 벡터 레지스터 세트인 **VREGs**로 구성됩니다.

**VREGs:** 각 TPU v5p 코어는 64개의 32비트 VREGs를 가집니다(TPU v4는 32개). 따라서 코어당 총 `64 * 8 * 128 * 4 = 256kB`의 VREG 메모리가 있으며, 칩 전체로는 두 개의 코어가 있으므로 이의 2배입니다. TPU v5p는 매 사이클마다 VMEM에서 3개의 레지스터를 로드하고, VMEM에 1개의 레지스터를 쓸 수 있습니다.

**VPU:** VPU는 `(8, 128)` 모양의 2D 벡터 산술 유닛으로, 128 차원은 레인 축(lane axis)이라 하고 8 차원은 서브레인 축(sublane axis)이라고 합니다. v5의 각 (레인, 서브레인) 쌍에는 서로 독립적인 4개의 표준 부동소수점 ALU가 포함되어 있습니다. VPU는 대부분의 산술 instruction을 각 ALU에서 한 사이클에 실행하며(vadd 또는 벡터 덧셈과 같이) 2 사이클의 지연 시간을 가집니다. 따라서 예를 들어 v5에서는 매 사이클마다 VREG에서 4쌍의 f32 값을 더할 수 있습니다. 일반적인 VPU instruction은 `{v2 = vadd.8x128.f32 v0, v1}`과 같이 보일 수 있으며, 여기서 v0과 v1은 입력 VREG이고 v2는 출력 VREG입니다.

모든 레인과 서브레인은 순수한 SIMD 방식으로 매 사이클마다 동일한 프로그램을 실행하지만, 각 ALU는 다른 연산을 수행할 수 있습니다. 따라서 예를 들어, 한 사이클에 1개의 vadd와 1개의 vsub를 처리할 수 있으며, 각각은 두 개의 전체 VREG에서 작동하고 출력을 세 번째에 씁니다.

**Pop Quiz [Calculating VPU throughput]:** 위의 정보를 사용하여 TPU v5p가 초당 몇 번의 벡터 FLOPs를 수행할 수 있는지 계산해 보세요. TPU v5p의 클럭 속도는 약 1.75GHz입니다.

*답*: 매 사이클마다 각 코어는 `8 * 128`개의 ALU에서 4개의 벡터 instruction을 실행할 수 있습니다. 이는 칩 전체에 대해 `8 * 128 * 4 * 2` FLOPs/cycle을 제공하며, 즉 `8 * 128 * 4 * 2 * 1.75e9 = 1.4e13 FLOPs/s`입니다. 이 값이 MXU FLOPs/s인 약 `2e14`(대략 10배 차이)보다 얼마나 작은지 주목하세요.

**Reductions(축소)):** 일반적으로 서브레인 차원에서의 통신이나 축소는 레인 차원보다 쉽습니다. 예를 들어, VPU는 약 한 사이클 만에 크기 8의 축을 따라 롤링할 수 있는 레인 내 셔플(intra-lane shuffle) 연산을 지원합니다. 이는 서브레인 차원을 따라 효율적인 축소를 수행하는 데 사용될 수 있습니다(단지 2, 4, 6만큼 셔플하고 3쌍의 elementwise sums을 수행하면 됨).

Cross-lane reductions(교차 레인 축소)는 훨씬 더 어렵고 XLU 또는 "cross lane unit"이라는 별도의 하드웨어 유닛을 포함하며, 이는 느리고 상당히 비쌉니다.

**GPU와의 비교:** NVIDIA GPU에 익숙한 분들을 위해 설명하자면, VPU의 각 ALU는 CUDA 코어와 유사하며, 단일 VPU 레인은 보통 32개의 CUDA 코어 세트로 구성된 SIMD 연산을 수행하는 "워프 스케줄러(Warp Scheduler)"와 유사합니다. 레인 내에서의 축소는 꽤 쉽지만, 레인을 넘어야 한다면 훨씬 느린 VMEM/XLU/SMEM을 최소 한 번 거쳐야 합니다.

### Scalar Core

Scalar Core(스칼라 코어)는 TPU의 제어 유닛입니다. 모든 instruction을 가져와 디스패치하고 HBM에서 VMEM으로의 전송을 실행하며, 스칼라 메타데이터 작업을 하도록 프로그래밍할 수 있습니다. 스칼라 코어는 단일 스레드이므로, 이의 한 가지 부작용은 TPU의 각 코어가 사이클당 하나의 DMA 요청만 생성할 수 있다는 것입니다.

이것을 맥락에 넣어보면, 단일 스칼라 코어는 VPU(4096개의 ALU로 구성), 4개의 MXU, 2개의 XLU 및 여러 DMA 엔진을 제어합니다. 단위 연산당 제어의 고도로 편향된 특성은 하드웨어 효율성의 원천이지만, 흥미로운 방식으로 데이터 종속 벡터화를 수행하는 능력을 제한하기도 합니다.

### Appendix B: All about GPUs

Volta 세대(V100) 이후로, TPU와 GPU는 매우 유사해 보이기 시작했습니다: _둘 다 행렬 곱셈을 매우 빠르게 수행하는 것을 목표로 합니다_. 둘 다 CPU에 부착된 가속기로 작동하며 많은 구성 요소가 대략적으로 유사합니다(모든 용어를 모른다고 걱정하지 마세요, 나중에 모두 소개할 것입니다):

|     TPU     |                    GPU                    |
| :---------: | :---------------------------------------: |
| Tensor Core |      SM ("Streaming Multiprocessor")      |
|     HBM     |                   DRAM                    |
|    VMEM     |     SMEM (often used as an L1 cache)      |
|     VPU     | Warp scheduler (a set of SIMD CUDA cores) |
|     MXU     |                Tensor Core                |
|     ICI     |              NVLink/NVSwitch              |

GPU의 핵심 유닛은 SM, 즉 "streaming multiprocessor(스트리밍 멀티프로세서)"이며, 이는 위에서 설명한 전체 TPU Tensor Core와 대략적으로 유사합니다. 하지만 TPU에 비해 GPU는 _훨씬_ 더 많은 SM을 가지고 있습니다(H100은 약 144개). 각 SM에는 TPU MXU처럼 작동하는 자체 행렬 곱셈 유닛(혼동스럽게도 Tensor Core라고 불림)과 TPU VPU처럼 작동하는 4개의 좁은 SIMD 유닛 세트(1024 레인 대신 32 레인)인 워프 스케줄러가 있습니다. 더 많은 독립적인 SM은 계산을 더 유연하게 만들지만(각각이 완전히 독립적인 작업을 할 수 있으므로), 하드웨어를 더 비싸고 추론하기 복잡하게 만듭니다.

{% include figure.liquid path="assets/img/b100-sm-diagram.png" class="img-small" caption="<b>Figure:</b> Blackwell (B100) SM의 기본 구성 요소. 다이어그램은 4개의 SIMD 연산 유닛(워프 스케줄러라고 부름)을 보여주며, 각각 행렬 곱셈을 위한 Tensor Core를 가지고 있습니다. 또한 워프 스케줄러별 레지스터, SM 수준의 L1 캐시, 그리고 Blackwell에서 새로 추가된 TMEM 또는 텐서 메모리를 보여줍니다." %}

각 SM에는 데이터 접근을 가속화하고 레지스터 spilling에 사용되는 O(256kB)의 L1 캐시(SMEM이라고도 함)가 있습니다. L1 캐시에 사용되는 메모리의 일부는 스레드 블록의 모든 스레드에서 접근할 수 있는 공유 메모리로 선언될 수도 있으며, user-defined(사용자 정의) 캐시, 병렬 축소 및 동기화 등에 사용됩니다(TPU의 VMEM과 유사).

GPU에는 모든 SM이 공유하는 추가적인 L2 캐시도 있습니다. VMEM과 달리, 이는 하드웨어 관리되며 캐시 적중률을 최적화하는 것이 종종 성능에 중요합니다.

**Networking:**

* 주요 차이점은 NVIDIA GPU가 일반적으로 스위치(NVLink $\rightarrow$ NVSwitch)를 통해 8-256 GPU의 'clique'에 있다는 것입니다. 이는 해당 'clique' 내의 모든 GPU 간의 점대점 통신을 허용하지만, 256개 이상의 통신은 훨씬 느리다는 것을 의미합니다. 이는 256개 이상에서의 훈련은 일반적으로 확장을 위해 더 복잡한 파이프라인 병렬 처리가 필요함을 의미합니다(대조적으로, PaLM은 각각 3072개의 TPU 칩으로 구성된 두 개의 clique에서 훈련되었습니다).
* AllReduce와 같은 일반적인 신경망 연산의 경우, all-to-all(전-대-전) 연결은 이점이 없습니다(어쨌든 동일한 통신 패턴이 발생해야 함). 하지만 더 많은 GPU에 MoE 모델을 저장하고 전문가(expert)를 더 효율적으로 전송할 수 있게 합니다.
* 각 GPU는 GPU 자체와 비슷한 비용의 스위치를 필요로 하므로, ICI와 같은 온칩 interconnect이 더 저렴합니다.
* [NVIDIA deep learning performance](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#gpu-arch) 
* [NVSwitch](https://www.nvidia.com/en-au/data-center/nvlink/) 
* 텐서 병렬 처리 / 파이프라인 병렬 처리 전환점이 매우 다릅니다!

### Appendix C: How does a systolic array work?

TPU MXU의 핵심에는 `128x128` 시스톨릭 배열(TPU v6e에는 `256x256`)이 있습니다. 완전히 포화 상태(fully saturated)일 때, 시스톨릭 배열은 8 클럭 사이클마다 `bfloat16[8,128] @ bf16[128x128] -> f32[8,128]`<d-footnote>이 표기법에 익숙하지 않다면, 이는 bfloat16 element를 가진 `8x128` 행렬을 bfloat16 element를 가진 `128x128` 행렬과 곱하고 그 결과를 float32 element를 가진 `8x128` 행렬에 저장하는 것을 의미합니다.</d-footnote> 곱셈을 한 번 수행할 수 있습니다.

* 핵심은, 시스톨릭 배열은 각각 곱셈과 덧셈 연산을 수행할 수 있는 2차원 `128x128` (`=16,384`)개의 ALU 그리드입니다.
* 가중치(**W**, `128x128` 입력)는 위에서 아래로(RHS라고 함) 전달되고, 입력(**X**, `8x128` 입력)은 왼쪽에서(LHS라고 함) 전달됩니다.

다음은 가중치 세트(파란색)를 활성화 세트(녹색)와 곱하는 간단한 애니메이션입니다. 가중치(RHS)가 먼저 부분적으로 대각선으로 로드된 다음, 활성화도 대각선으로 공급되는 것을 알 수 있습니다. 아래 각 프레임에서, 우리는 모든 겹치는 녹색과 파란색 유닛을 곱하고, 그 결과를 위에서 전달된 residual과 합한 다음, 그 결과를 차례로 한 유닛 아래로 전달합니다.

{% include figure.liquid path="assets/img/systolic-array.gif" %}

다음은 출력이 계산에서 스트리밍되는 것을 보여주는 이 애니메이션의 더 일반적인 버전입니다:

{% include figure.liquid path="assets/img/systolic-array2.gif" class="img-small" %}

다음은 여러 RHS 및 LHS 배열에 걸쳐 어떻게 파이프라인화될 수 있는지를 보여주는 다이어그램입니다:

{% include figure.liquid path="assets/img/systolic-array-pipelining.png" class="img-fluid" %}

가중치(RHS)와 활성화(LHS)가 로드될 때 초기 파이프라인 버블이 있습니다. 그 초기 버블 이후에는 추가적인 버블 없이 새로운 입력과 가중치를 로드할 수 있습니다.


다음은 bf16[2, 3] x bf16[3, 3] 행렬 곱셈의 좋지 않은 애니메이션으로, 배치 1과 크기 3의 입력 활성화와 2x3 가중치 행렬의 matmul로 상상할 수 있습니다. 이것은 이전 슬라이드와 비교하여 회전되어 있으며 입력은 아래가 아닌 오른쪽으로 흐르지만, 대략적인 구조를 볼 수 있습니다.

{% include figure.liquid path="assets/img/systolic-array-bad.gif" class="img-small" %}

우리는 너무 큰 파이프라인 버블 없이 큰 행렬을 곱하기 위해 이것을 효율적으로 파이프라인화할 수 있습니다. 그렇긴 하지만, 우리 행렬의 모양이 MXU의 측면 차원(일반적으로 128x128)보다 큰 것이 중요합니다. 일부 TPU(TPU v3 이후)에는 여러 개의 MXU가 있으므로(TPU v3의 경우 2개, TPU v4/5의 경우 4개), 타일링 차원이 128 * MXU 수보다 큰지 확인해야 합니다. [여기](https://www.youtube.com/watch?v=sJltBQ4MOHA)에 이것을 위한 좋은 애니메이션이 있습니다.

Trillium (TPU v6e)는 `256x256` 시스톨릭 배열을 가지고 있어, 사이클당 4배 더 많은 FLOPs를 수행할 수 있습니다. 이는 또한 MXU를 완전히 활용하기 위해 텐서의 차원이 두 배 더 커져야 함을 의미합니다.

[블로그 게시물](https://fleetwood.dev/posts/domain-specific-architectures#google-tpu) 고정된 가중치 행렬에 대한 시스톨릭 배열 곱셈의 또 다른 훌륭한 애니메이션이 있습니다.[역자 주: [해당 글](https://tulip-phalange-a1e.notion.site/AI-Domain-Specific-Architectures-23cc32470be28025bbffe42e15e58d99?pvs=74)의 번역본입니다, 다만 애니메이션은 원문에서 보시는게 좋습니다!]
