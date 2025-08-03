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
  - name: Worked Problems
  - name: Appendix
  - subsections:
    - name: "Appendix A: All about GPUs"
    - name: "Appendix B: How does a systolic array work?"
    - name: "Appendix C: More on TPU internals"

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

* **MXU** (Matrix Multiply Unit)는 TensorCore의 핵심입니다. 대부분의 TPU 세대에서, MXU는 시스톨릭 배열(systolic array)을 사용하여 8 사이클마다 `bfloat16[8,128] @ bf16[128,128] -> f32[8,128]` 행렬 곱셈을 한 번 수행합니다

* **MXU** (Matrix Multiply Unit)는 TensorCore의 핵심입니다. 대부분의 TPU 세대에서, MXU는 시스톨릭 배열(systolic array)을 사용하여 8 사이클마다 `bfloat16[8,128] @ bf16[128,128] -> f32[8,128]` 행렬 곱셈<d-footnote>TPU v6e (Trillium)는 256x256 MXU를 사용하며, 이전 세대는 모두 128x128을 사용합니다.</d-footnote> 한 번 수행합니다 (<a href="#appendix-b-how-does-a-systolic-array-work">Appendix B</a> 참조).  
  * 이는 TPU v5e에서 1.5GHz로 작동할 때 MXU당 약 `5e13` bf16 FLOPs/s에 해당합니다. 대부분의 TensorCore에는 2개 또는 4개의 MXU가 있으므로, 예를 들어 TPU v5e의 총 bf16 FLOPs/s는 `2e14`입니다.  
  * TPU는 또한 더 높은 처리량을 가진 더 낮은 정밀도의 matmul도 지원합니다 (예: 각 TPU v5e 칩은 `4e14` int8 OPs/s를 수행할 수 있습니다).

* **VPU** (Vector Processing Unit)는 ReLU 활성화나 벡터 간의 원소별 덧셈 또는 곱셈과 같은 일반적인 수학 연산을 수행합니다. Reduction(sums) 연산도 여기서 수행됩니다. 자세한 내용은 <a href="#appendix-c-tpu-internals">Appendix C</a> 를 참조하시면 됩니다. 
* **VMEM** (Vector Memory)은 TensorCore 내부에 위치한 온칩(on-chip) 스크래치패드로, 연산 유닛에 근접해있습니다. HBM보다 훨씬 작지만(예: TPU v5e에서는 128MiB) MXU와의 대역폭은 훨씬 높습니다. VMEM은 CPU의 L1/L2 캐시와 꽤나 유사하게 작동하지만, 훨씬 크고 프로그래머가 제어(programmer-controlled)할 수 있습니다. HBM의 데이터는 TensorCore가 계산을 수행하기 전에 VMEM으로 복사되어야 합니다.

**TPU는 행렬 곱셈이 아주, 아주 빠릅니다**. TPU의 주요 업무이기도 하고 잘 하기도 합니다. 지금까지 가장 강력한 TPU 중 하나인 [TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture)는 코어당 초당 `2.5e14` bf16 FLOPs / second 또는 칩당 `5e14` bf16 FLOPs / second 을 수행할 수 있습니다. 8960개 칩으로 구성된 단일 pod는 초당 4 exaflops를 처리할 수 있습니다. 이는 *어마어마한* 양입니다. 이는 세계에서 가장 강력한 슈퍼컴퓨터 중 하나이며, 구글은 이를 다수 보유하고 있습니다.<d-footnote>TPU와 특히 이의 시스톨릭 배열(systolic arrays)이 이토록 강력한 하드웨어 가속기인 이유는, 행렬 곱셈이 $O(n^2)$ 바이트에 대해 $O(n^3)$의 연산을 사용하는 몇 안 되는 알고리즘 중 하나이기 때문입니다. 이로 인해 일반적인 ALU가 메모리 대역폭이 아닌 연산 자체에 의해 병목 현상을 겪기 매우 쉽습니다.</d-footnote>

위의 다이어그램에는 제어 흐름 처리(control flow handling)에 사용되는 SMEM 및 스칼라(scalar) 유닛과 같은 몇 가지 다른 구성 요소도 포함되어 있으며, 이는 <a href="#appendix-c-tpu-internals">Appendix C</a>에서 짧게 다루지만, 꼭 이해하셔야 하지는 않습니다. 반면에 HBM은 중요하면서 또한 비교적 간단합니다:

* **HBM** (High Bandwidth Memory) 은 TensorCore에서 사용할 텐서를 저장하는 큰 용량의 빠른 메모리입니다. HBM은 보통 수십 기가바이트의 용량을 가집니다(예를 들자면, [TPU v5e는 16GiB의 HBM을 가짐](https://cloud.google.com/tpu/docs/v5e#system_architecture)).

  * 계산이 필요할 때, 텐서는 HBM에서 VMEM(아래 예제 있음)을 통해 스트리밍되어 MXU로 들어가고, 결과는 VMEM에서 다시 HBM으로 쓰입니다.
  
  * HBM과 TensorCore(VMEM을 통해) 간의 대역폭은 "HBM 대역폭” (보통 1-2TB/sec)이라 하며, 메모리 병목(memory-bound) 워크로드에서 계산이 얼마나 빨리 수행할 수 있는지의 제약 사항이 됩니다. 

**보통 모든 TPU 연산은 파이프라인화되고 중첩됩니다.** matmul $X \cdot A \to Y$ 를 수행하기 위해, TPU는 먼저 HBM에서 $A$ 와 $X$ 행렬의 청크를 VMEM으로 복사한 다음, 이를 MXU로 로드하여 8x128($X$의 경우) 및 128x128($A$의 경우) 청크를 곱하고, 그 결과를 청크 단위로 다시 HBM에 복사합니다. 이를 효율적으로 수행하기 위해, matmul은 VMEM으로/에서 복사하는 작업이 MXU 작업과 중첩되도록 파이프라인화됩니다. 이를 통해 MXU는 메모리 전송을 기다리지 않고 계속 작동할 수 있으며, matmul이 메모리 병목이 아닌 연산 병목 상태를 유지하게 합니다.

다음은 HBM에서 원소별 곱셈(elementwise product)을 수행하는 방법의 예제입니다:

{% include figure.liquid path="assets/img/pointwise-product.gif" caption="<b>Figure:</b> HBM에서 바이트를 로드하여 TPU에서 원소별 곱셈(pointwise product)을 수행하는 애니메이션. 바이트가 메모리에서 청크 단위로 스트리밍되고, 전체 배열이 구체화되기를 기다리지 않고 부분 결과가 파이프라인으로 다시 전송되는 방식을 주의 깊게 봐주세요." %}

A matmul would look nearly identical except it would load into the MXU instead of the VPU/Vector unit, and the loads and stores would occur in a different order, since the same weight chunk is used for multiple chunks of activations. You can see chunks of data streaming into VMEM, then into the VREGs (vector registers), then into the Vector Unit, then back into VMEM and HBM. As we're about to see, if the load from HBM to VMEM is slower than the FLOPs in the Vector Unit (or MXU), we become "bandwidth bound” since we're starving the VPU or MXU of work.

<p markdown=1 class="takeaway">**Key takeaway:** TPUs are very simple. They load weights from HBM into VMEM, then from VMEM into a systolic array which can perform around 200 trillion multiply-adds per second. The HBM $\leftrightarrow$ VMEM and VMEM $\leftrightarrow$ systolic array bandwidths set fundamental limits on what computations TPUs can do efficiently.</p>

**VMEM and arithmetic intensity:** VMEM is much smaller than HBM but it has a much higher bandwidth to the MXU. As we saw in [Section 1](../roofline), this means if an algorithm can fit all its inputs/outputs in VMEM, it's much less likely to hit communication bottlenecks. This is particularly helpful when a computation has poor arithmetic intensity: VMEM bandwidth is around 22x higher than HBM bandwidth which means an MXU operation reading from/writing to VMEM requires an arithmetic intensity of only 10-20 to achieve peak FLOPs utilization. That means if we can fit our weights into VMEM instead of HBM, our matrix multiplications can be FLOPs bound at much smaller batch sizes. And it means algorithms that fundamentally have a lower arithmetic intensity can still be efficient. VMEM is just so small this is often a challenge.<d-footnote>We sometimes talk about VMEM prefetching, which refers to loading weights ahead of time in VMEM so we can mask the cost of loading for our matmuls. For instance, in a normal Transformer we can sometimes load our big feed-forward weights into VMEM during attention, which can hide the cost of the weight load if we're memory bandwidth bound. This requires our weights to be small enough or sharded enough to fit a single layer into VMEM with space to spare.</d-footnote>

{% include figure.liquid path="assets/img/tpu-bandwidth.png" class="img-fluid" %}

**A TPU chip typically (but not always) consists of two TPU cores which share memory and can be thought of as one large accelerator** with twice the FLOPs (known as a "megacore" configuration). This has been true since TPU v4. Older TPU chips have separate memory and are regarded as two separate accelerators (TPU v3 and older). Inference-optimized chips like the TPU v5e only have one TPU core per chip.

{% include figure.liquid path="assets/img/cores.png" class="img-fluid img-small" %}

**Chips** are arranged in **sets of 4 on a ‘tray'** connected to a **CPU host via PCIe network.**  This is the format most readers will be familiar with, 4 chips (8 cores, though usually treated as 4 logical megacores) exposed through Colab or a single TPU-VM. For inference chips like the TPU v5e, we have 2 trays per host, instead of 1, but also only 1 core per chip, giving us 8 chips = 8 cores.<d-footnote>On Cloud TPU VMs, each tray is exposed as part of a separate VM, so there are once again 4 cores visible.</d-footnote>

{% include figure.liquid path="assets/img/pcie.png" class="img-fluid" %}

**PCIe bandwidth is limited:** Like the HBM $\leftrightarrow$ VMEM link, the CPU $\leftrightarrow$ HBM PCIe connection has a specific bandwidth that limits how quickly you can load from host memory to HBM or vice-versa. PCIe bandwidth for TPU v4 is 16GB / second each way, for example, so close to 100x slower than HBM. We *can* load/offload data into the host (CPU) RAM, but not very quickly.

## TPU Networking

**Chips are connected to each other through the ICI network in a Pod**. In older generations (TPU v2 and TPU v3), inference chips (e.g., TPU v5e), and Trilium (TPU v6e), ICI ("inter-chip interconnects”) connects the 4 nearest neighbors (with edge links to form a 2D torus). TPU v4 and TPU v5p are connected to the nearest 6 neighbors (forming a 3D torus). Note these connections do **not** go through their hosts, they are direct links between chips.

{% include figure.liquid path="assets/img/ici-wraparound.png" class="img-fluid img-small" %}

The toroidal structure reduces the maximum distance between any two nodes from $N$ to $N / 2$, making communication much faster. TPUs also have a "twisted torus” configuration that wraps the torus in a Mobius-strip like topology to further reduce the average distance between nodes.

**TPU pods (connected by ICI) can get really big:** the maximum pod size (called a **superpod**) is `16x16x16` for TPU v4 and `16x20x28` for TPU v5p. These large pods are composed of reconfigurable cubes of `4x4x4` chips connected by [optical wraparound links](https://arxiv.org/pdf/2208.10041)<d-footnote>The optical switch is simply a reconfigurable connection with the same ICI bandwidth. It just lets us connect cubes while retaining a wraparound link.</d-footnote> that we can reconfigure to connect very large topologies.

{% include figure.liquid path="assets/img/tpu-rack.png" class="img-fluid" %}

Smaller topologies (e.g. `2x2x1`, `2x2x2`) can also be requested, albeit with no wraparounds. This is an important caveat, since it typically doubles the time of most communication. Any multiple of a full cube (e.g. `4x4x4` or `4x4x8`) will have wraparounds provided by the optical switches.<d-footnote>Note that a `2x2x4` won't have any wraparounds since they are provided by the optical switches which are only available on a full cube. A TPU v5e 8x16 _will_ have a wraparound on the longer axis, however, since it doesn't use reconfigurable optical networking.</d-footnote>

{% include figure.liquid path="assets/img/subslices.png" class="img-fluid" %}

TPU v5e and Trillium pods consist of a single `16x16` 2D torus with wraparounds along any axis of size 16 (meaning an `8x16` has a wraparound on the long axis). TPUs v5e and v6e (Trillium) cannot expand beyond a 16x16 torus but pods can still communicate with each other over standard data-center networking (DCN), which connects TPU hosts to each other. Again, smaller topologies can be requested without wraps on dims $<16$.

{% include figure.liquid path="assets/img/more-subslices.png" class="img-fluid" %}

**This nearest-neighbor connectivity is a key difference between TPUs and GPUs**. GPUs are connected with a hierarchy of switches that approximate a point-to-point connection between every GPU, rather than using local connections like a TPU. Typically, GPUs within a node (8 GPUs for H100 or as many as 500 for B200) are directly connected, while larger topologies require O(log(N)) hops between each GPU. On the one hand, that means GPUs can send arbitrary data within a node in a single low-latency hop. On the other hand, TPUs are dramatically cheaper (since NVLink switches are expensive) and simpler to wire together, and can scale to much larger topologies because the number of links per device and the bandwidth per device is constant.

**ICI is very fast relative to DCN, but is still slower than HBM bandwidth.** For instance, a [TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture) has:

* `2.5e12` bytes/s (2.5 TB/s) of HBM bandwidth per chip.
* `9e10` bytes/s (90<d-footnote>The page above lists 100 GB/s of bandwidth, which is slightly different from what's listed here. TPU ICI links have slightly different bandwidths depending on the operation being performed. You can generally use the numbers in this doc without worry.</d-footnote> GB/s) of ICI bandwidth per axis, with 3 axes per chip. 
* `2.5e10` bytes/s (25 GB/s) of DCN (egress) bandwidth per host. Since we typically have 8 TPUs per host, this is really closer to `3.1e9` bytes / s / chip.

This means that when we split models across multiple chips, we need to be careful to avoid bottlenecking the MXU with slower cross-device communication.

**Multi-slice training:** A set of ICI-connected TPUs is called a **slice**. Different slices can be connected between each other using DCN, for instance to link slices on different pods. Since DCN is a much slower connection than ICI, one should try to limit how much our computation has to wait for data from DCN. DCN is host-to-host, so to transfer buffers from TPU to TPU over DCN, we first need to transfer over PCIe to the host, then egress over the network, then ingress over the target host network, then over PCIe into HBM.

## Key Takeaways

* TPUs are simple and can in most cases be thought of as a matrix multiply unit connected to memory (super fast), other chips over ICI (rather fast), and the rest of the datacenter over DCN (somewhat fast).

* Communication is limited by our various network bandwidths in order of speed: 
  * HBM bandwidth: Between a TensorCore and its associated HBM. 
  * ICI bandwidth: Between a TPU chip and its nearest 4 or 6 neighbors. 
  * PCIe bandwidth: Between a CPU host and its associated tray(s) of chips.
  * DCN bandwidth: Between multiple CPU hosts, typically hosts not connected by ICI.

* **Within a slice, TPUs are only connected to their nearest neighbors via ICI.** This means communication over ICI between distant chips in a slice needs to hop over the intervening chips first.

* **Weight matrices need to be padded to at least size 128** (256 on TPU v6) in both dimensions to fill up the MXU (in fact, smaller axes are padded to 128).

* **Lower precision matrix multiplication tends to be faster.** TPUs can do int8 or int4 FLOPs roughly 2x/4x faster than bfloat16 FLOPs for generations that support it. VPU operations are still performed in fp32.

* To avoid bottlenecking the TPU compute unit, we need to **make sure the amount of communication across each channel is proportional to its speed**.

* **Here are some specific numbers for our chips:**

| Model                                      | Pod size | Host size | HBM capacity/chip | HBM BW/chip (bytes/s) | FLOPs/s/chip (bf16) | FLOPs/s/chip (int8) |
| :----------------------------------------- | :------: | :-------: | :---------------: | :-------------------: | :-----------------: | :-----------------: |
| <span class="nowrap-header">TPU v3</span>  |  32x32   |    4x2    |       32GB        |        9.0e11         |       1.4e14        |       1.4e14        |
| <span class="nowrap-header">TPU v4p</span> | 16x16x16 |   2x2x1   |       32GB        |        1.2e12         |       2.75e14       |       2.75e14       |
| <span class="nowrap-header">TPU v5p</span> | 16x20x28 |   2x2x1   |       96GB        |        2.8e12         |       4.59e14       |       9.18e14       |
| <span class="nowrap-header">TPU v5e</span> |  16x16   |    4x2    |       16GB        |        8.1e11         |       1.97e14       |       3.94e14       |
| <span class="nowrap-header">TPU v6e</span> |  16x16   |    4x2    |       32GB        |        1.6e12         |       9.20e14       |       1.84e15       |

Host size refers to the topology of TPUs connected to a single host (e.g. TPU v5e has a single CPU host connected to 8 TPUs in a 4x2 topology). Here are interconnect figures:

| Model       | ICI BW/link (one-way, bytes/s) | ICI BW/link (bidi, bytes/s) |
| :---------- | :----------------------------: | :-------------------------: |
| **TPU v3**  |              1e11              |            2e11             |
| **TPU v4p** |             4.5e10             |            9e10             |
| **TPU v5p** |              9e10              |           1.8e11            |
| **TPU v5e** |             4.5e10             |            9e10             |
| **TPU v6e** |              9e10              |           1.8e11            |

We include both one-way (unidirectional) bandwidth and bidi (bidirectional) bandwidth since unidirectional bandwidth is more true to the hardware but bidirectional bandwidth occurs more often in equations involving a full ring.<d-footnote>By bidi (bidirectional) bandwidth we mean the total bytes that can be sent along a single link in both directions, or equally, the total number of outgoing bytes from a single TPU along a particular axis, assuming we can use both links efficiently. This is true when we have a functioning ring, AKA when we have a wraparound connection on the particular axis. This occurs on inference chips when we have a full 16 axis, or on training chips (v*p) when we have an axis which is a multiple of 4. We prefer to use the bidirectional bandwidth because it appears frequently in calculations involving bidirectional comms.</d-footnote>

PCIe bandwidth is typically around `1.5e10` bytes / second per chip<d-footnote>Trillium (TPU v6e) has 32GB/s, about 2x higher than v5.</d-footnote>, while DCN bandwidth is typically around `2.5e10` bytes / second per host. We include both unidirectional and bidirectional bandwidth for completeness. Typically bidirectional bandwidth is the more useful number when we have access to a full wraparound ring, while one-way bandwidth is more true to the hardware.

## Worked Problems

These numbers are a little dry, but they let you make basic roofline estimates for model performance. Let's work a few problems to explain why this is useful. You'll see more examples in Part 3.

**Question 1 [bounding LLM latency]:** Say you want to sample from a 200B parameter model in bf16 that's split across 32 TPU v4p. How long would it take to load all the parameters from HBM into the systolic array? *Hint: use the numbers above.*

{% details Click here for the answer. %}

**Answer:** We're loading `sizeof(bf16) * 200e9 = 400e9` bytes on 32 chips, meaning 12.5e9 bytes / chip, each with an HBM bandwidth of 1.23e12. So the load takes around 10ms.

That's pretty cool, because *that's a reasonable lower bound on the latency of sampling* from the model. Each sampling step needs to load all parameters from HBM, so it cannot take less than 10 ms. In practice, at small batch sizes, this is close to being achievable.

{% enddetails %}

**Question 2 [TPU details]:** Consider a full TPU v5e pod. How many total CPU hosts are there? How many TPU TensorCores? What is the total FLOPs/s for the whole pod? What is the total HBM? Do the same exercise for TPU v5p pod.

{% details Click here for the answer. %}

**Answer:** For TPU v5e, each pod is `16x16` and each host is a 4x2 slice, so we have `16*16 / 8 = 32` hosts. For TPU v5e, each TPU has only one core, so we have 256 TensorCores. The total FLOPs/s is `16*16*2e14 = 5.1e16` in bfloat16. Each chip has 16GB of HBM, so that's `256 * 16 = 4TB` of memory.

For a full TPU v5p pod, we have `16x20x28` chips and each host is 2x2x1, so we have `16*20*28 / 2*2 = 2,240` hosts. For TPU v5p, each TPU has two TensorCores, so we have `8960 * 2 = 17,920` cores. The total FLOPs/s is `8960 * 4.5e14 = 4e18` in bfloat16. Each chip has 96GB of HBM, so that's `8960 * 96 = 860TB` of memory.

{% enddetails %}

**Question 3 [PCIe operational intensity]:** Imagine we're forced to store a big weight matrix $A$ of type $\text{bfloat16}[D, F]$, and a batch of activations $x$ of type $\text{bfloat16}[B, D]$ in host DRAM and want to do a matrix multiplication on them. This is running on a single host, and we're using a single TPU v6e chip attached to it. You can assume $B \ll D$, and $F = 4D$ (we'll see in future chapters why these are reasonable assumptions). What is the smallest batch size $B$ we need to remain FLOPs bound over PCIe? Assume PCIe bandwidth of 1.5e10 bytes / second.

{% details Click here for the answer. %}

**Answer:** We have to perform $2BDF$ floating point operations, and each chip can perform `9.2e14` floating point operations per second. This then requires $2BDF / 9.2e14$ seconds to perform. We have to load $2DF + 2BD$ bytes from DRAM, and write $2BF$ bytes back to it. We are bottlenecked by PCIe transfer speeds, so we need $2 \cdot (BD + DF + BF) / 1.5e10$ seconds to transfer data to and from the TPU. Since we want computation to take longer than weight loading, assuming we can overlap all weight loading with computation, we want $2BDF / 9.2e14 > 2 \cdot (BD + DF + BF) / 1.5e10$. We can simplify this using our assumptions that $B \ll D$, and $F = 4D$, to get

$$\frac{8BD^2}{9.2e14} > \frac{8D^2}{1.5e10}$$

or

$$B > \frac{9.2e14}{1.5e10} \simeq 61,000$$

{% enddetails %}

**Question 4 [general matmul latency]:** Let's say we want to multiply a weight matrix int8[16384, 4096] by an activation matrix of size int8[B, 4096] where B is some unknown batch size. Let's say we're on 1 TPUv5e to start.

1. How long will this multiplication take as a function of B? *Hint: it may help to calculate how long it will take to load the arrays from HBM and how long the multiplication will actually take. Which is bottlenecking you?* 
2. What if we wanted to run this operation out of VMEM? How long would it take as a function of B?

{% details Click here for the answer. %}

**Answer:** (1) The number of floating point operations we need to perform is $2 \cdot 4096 \cdot 16384 \cdot B = 1.3e8 \cdot B$. So $T_{\text{math}} = (1.3e8 \cdot B) / 3.94e14$ seconds. We need to load $16384 \cdot 4096 + 4096 \cdot B$ bytes from HBM to VMEM, and write back $16384 \cdot B$ bytes from VMEM to HBM. This means $T_{\text{comms}} = (6.7e7 + 2e4\cdot B) / 8.1e11$ seconds. Assuming as much overlap of communication and computation as possible, the whole multiplication will take approximately 

$$\max\{T_{\text{math}}, T_{\text{comms}}\} = \max\left\{\frac{6.7e7 + 2e4\cdot B}{8.1e11}, \frac{1.3e8 \cdot B}{3.94e14}\right\}$$

We'll be FLOPs-bound when $\frac{6.7e7 + 2e4\cdot B}{8.1e11} < \frac{1.3e8 \cdot B}{3.94e14}$, or equivalently, $B > 271$. This is slightly larger than the 240 number we derive below because we factor in the full impact of $$D$$ and $$F$$. 

(2) If instead we are loading from VMEM, let's consider VMEM bandwidth to the MXU as 22 times the HBM $\leftrightarrow$ VMEM bandwidth. This turns our data loading denominator from 8.1e11 to 1.78e13, and we get $B > 11$. Note that in practice, we cannot dedicate all of our VMEM bandwidth to loading $W$, so in practice it will be closer to 20.

{% enddetails %}

**Question 5 [ICI bandwidth]:** Let's say we have a TPU v5e `4x4` slice. Let's say we want to send an array of type `bfloat16[8, 128, 8192]` from `TPU{0,0}` to `TPU{3, 3}`. Let's say the per-hop latency for TPU v5e is $1\mu s$. 

1. How soon will the first byte arrive at its destination?
2. How long will the total transfer take?

{% details Click here for the answer. %}

**Answer:** In a TPUv5e we have 2D connectivity. Because we have only a `4x4` slice (with no axes of size 16), we have no wraparound connections. Thus there are two ports from which our target chip can receive data, and likewise two ports from which our source chip can send data. The amount of data we have to transfer is `2 * 8 * 128 * 8192 = 1.7e7` bytes. We can transfer from both ports simultaneously (i.e. send half the array right and half down), so we get `2 * 4.5e10 = 9e10` bytes transferred per second, which means it'll take about `1.7e7 / 9e10 = 188us` to transfer the whole array through (assuming we're bandwidth bound). In a `4x4` slice, we have six hops between chips $(0, 0)$ and $(3, 3)$, since there are no wraparound links for axes with fewer than 16 chips. Since the latency of each hop is about $1\mu s$, the first byte will arrive in about`6us` and the total transfer will take `188us`.

{% enddetails %}

**Question 6 [pulling it all together, hard]:** Imagine you have a big matrix **A**: `int8[128 * 1024, 128 * 1024]` sharded evenly across a TPU v5e 4x4 slice but offloaded to host DRAM on each chip. Let's say you want to copy the entire array to TPU{0, 0} and multiply it by a vector `bf16[8, 128 * 1024]`. How long will this take? *Hint: use the numbers above.*

{% details Click here for the answer. %}

**Answer:** Let's start by outlining the operations we have to perform. Our array is about 16GB. From the table above, a TPU v5e host has a 4x2 topology, so a 4x4 has 2 hosts, Thus, since our array is evenly sharded, each host effectively contains a chunk of 1/2 of the array, or 8GB. We need to copy these chunks all to TPU{0,0}, which gives us two options:

1. We can copy over DCN and then load the entire unsharded array over PCIe into HBM. 
2. We can load our sharded arrays onto their corresponding TPUs, then perform a gather over ICI, then perform the matmul on TPU{0,0}.

It should be clear that option (2) is better. DCN is slow compared to ICI and we'd much prefer to load a big array over many PCIe links rather than just a few (the 8 on host 0). Here's a diagram of part of the system. As described above, note that TPUs are connected to their neighbors by ICI (even across hosts), all TPUs are connected to their host CPU (via PCIe), and hosts are connected by DCN.

{% include figure.liquid path="assets/img/challenge-problem.png" class="img-fluid img-small" caption="Each chip actually has its own PCIe link to its host, though for clarity only one is shown here." %}

Now let's work through how long each piece will take:

1. **PCIe load**: we're loading chunks of 16GB / 2 = 8GB over 16 PCIe links, each of which has `1.5e10` bytes/second bandwidth. Thus this will take about 33ms.

2. **ICI copy:** each TPU now has 16GB / 16 = 1GB of our array. Our ICI bandwidth is 9e10 bytes/second per link *bidirectional*, and you'll notice from the above diagram that only 2 of the 4 ICI links on the TPU v5e are in use in this topology for TPU{0,0}. Since TPU{0,0} needs to receive a total of 15GB along 2 axes at `4.5e10` bytes/s/link, we can lower bound the time by `15e9 / (4.5e10 * 2) = 167ms`. In practice this probably isn't achievable because the load is very uneven, but it's probably within a factor of 2. As you'll see in Section 2, performing a full AllGather would also take roughly `16e9 / (4.5e10 * 2)`, so this is close to optimal.

3. **HBM $\rightarrow$ MXU load:** to perform our final matmul, we need to load these 16e9 bytes plus the bf16[8, 128 \* 1024] array (another 2MB, so negligible) over HBM bandwidth into the MXU, which will take `16e9 / 8.1e11 = 19ms`.

4. **FLOPs:** we're performing a total of $$2 \cdot 8 \cdot 128 \cdot 1024 \cdot 128 \cdot 1024 = 2.7e11$$ FLOPs, and since we can perform `1.97e14` bf16 FLOPs/s, we get 1.3ms.

An upper bound for the total time is the sum of all of these times, but since the TPU can typically overlap these operations, we can think of this as a pipelining problem that's bottlenecked by the slowest piece. Assuming that's true, then the answer is about 150-200ms.

{% enddetails %}

<h3 markdown=1 class="next-section">That's it for Part 2! For Part 3, covering partitioning and cross-TPU communication, [click here](../sharding).</h3>

## Appendix

### Appendix A: All about GPUs

Since the Volta generation (V100), TPUs and GPUs have started to looked a lot alike: _they both aim to do matrix multiplication very fast_. They both act as an accelerator attached to a CPU and many components are roughly analogous (don't worry if you don't know all the terminology, we'll introduce them all later):

|     TPU     |                    GPU                    |
| :---------: | :---------------------------------------: |
| Tensor Core |      SM ("Streaming Multiprocessor")      |
|     HBM     |                   DRAM                    |
|    VMEM     |     SMEM (often used as an L1 cache)      |
|     VPU     | Warp scheduler (a set of SIMD CUDA cores) |
|     MXU     |                Tensor Core                |
|     ICI     |              NVLink/NVSwitch              |

The core unit of a GPU is an SM, or "streaming multiprocessor", which is roughly analogous to the whole TPU Tensor Core described above. Compared to TPUs, though, GPUs have _many_ more of them (an H100 has about 144). Each SM has its own matrix multiplication unit, confusingly called a Tensor Core, which acts like the TPU MXU, and a set of 4 narrow SIMD units called Warp schedulers that act like the TPU VPUs (with 32 lanes instead of 1024). More independent SMs makes computation more flexible (since each can do totally independent work) but also makes the hardware more expensive and complex to reason about. 

{% include figure.liquid path="assets/img/b100-sm-diagram.png" class="img-small" caption="<b>Figure:</b> the basic components of a Blackwell (B100) SM. The diagram shows 4 SIMD compute units (which we call warp schedulers), each with a Tensor Core for matrix multiplication. This also shows per-warp scheduler registers, an SM-level L1 cache, and TMEM or tensor memory which is a new addition in Blackwell." %}

Each SM also has an O(256kB) L1 cache (also called SMEM) used to speed data access and for register spilling. A section of the memory used for the L1 cache can also be declared as shared memory allowing access from any thread in the thread-block, and is used for user-defined caches, parallel reductions and synchronization, etc. (similar to VMEM on a TPU).

GPUs also have an additional L2 cache that is shared by all SMs. Unlike VMEM, this is hardware managed and optimizing cache hits is often important for performance.

**Networking:**

* Primary difference is that NVIDIA GPUs are typically in ‘cliques' of 8-256 GPUs via switches (NVLink $\rightarrow$ NVSwitch), which allow for point-to-point communication between any GPU within that ‘clique', but that means communication between more than 256 is significantly slower - this means training on more than 256 typically requires pipeline parallelism to scale, which is more complex (by contrast, PaLM was trained on two cliques of 3072 TPU chips each). 
* For common neural net operations such as AllReduce, all-to-all connections do not hold an advantage (as the same communication patterns must occur regardless), but it does allow for storing MoE models across more GPUs and transmitting the experts around more efficiently.
* Each GPU requires a switch that costs similar to the GPU itself, making on chip interconnect like ICI cheaper.
* [NVIDIA deep learning performance](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#gpu-arch) 
* [NVSwitch](https://www.nvidia.com/en-au/data-center/nvlink/) 
* Very different Tensor Parallelism / Pipeline Parallelism transition point!

### Appendix B: How does a systolic array work?

At the core of the TPU MXU is a `128x128` systolic array (`256x256` on TPU v6e). When fully saturated the systolic array can perform one `bfloat16[8,128] @ bf16[128x128] -> f32[8,128]`<d-footnote>If you are not familiar with this notation, it means: multiplying a `8x128` matrix with bfloat16 elements by a `128x128` matrix with bfloat16 elements and storing the results in a `8x128` matrix with float32 elements.</d-footnote> multiplication per 8 clock cycles.

* At its core, the systolic array is a 2D `128x128` (`=16,384`) grid of ALUs each capable of performing a multiply and add operation. 
* Weights (**W**, the `128x128` input) are passed down from above (called the RHS) while inputs (**X**, the `8x128` input) are passed in from the left (called the LHS).

Here is a simplified animation of multiplying a set of weights (blue) with a set of activations (green). You'll notice that the weights (RHS) are partially loaded first, diagonally, and then the activations are fed in, also diagonally. In each frame below, we multiply all the overlapped green and blue units, sum the result with any residual passed in from above, and then pass the result in turn down one unit.

{% include figure.liquid path="assets/img/systolic-array.gif" %}

Here's a more general version of this animation showing the output being streamed out of computation:

{% include figure.liquid path="assets/img/systolic-array2.gif" class="img-small" %}

Here's a diagram showing how this can be pipelined across multiple RHS and LHS arrays:

{% include figure.liquid path="assets/img/systolic-array-pipelining.png" class="img-fluid" %}

There is an initial pipeline bubble as the weights (RHS) and activations (LHS) are loaded. After that initial bubble, new inputs and weights can be loaded in without an additional bubble.

Here's a bad animation of a bf16[2, 3] x bf16[3, 3] matrix multiplication, which you could imagine as a matmul of a 2x3 weight matrix with an input activation of batch 1 and size 3. This is rotated compared to the previous slides and inputs flow out to the right instead of down, but you can roughly see the structure.

{% include figure.liquid path="assets/img/systolic-array-bad.gif" class="img-small" %}

We can efficiently pipeline this to multiply large matrices without too large a pipeline bubble. With that said, it's important that our matrices have shapes larger than the side dimension of the MXU, which is generally 128x128. Some TPUs (since TPU v3) have multiple MXUs, either 2 for TPU v3 and 4 for TPU v4/5, so we need to ensure tiling dimensions are larger than 128 * number of MXUs. [Here's](https://www.youtube.com/watch?v=sJltBQ4MOHA) a good animation for this.

Trillium (TPU v6e) has a `256x256` systolic array, which means it can perform 4x more FLOPs / cycle. This also means the dimensions of your tensors needs to be twice as large to utilize the MXU fully.

[This blog post](https://fleetwood.dev/posts/domain-specific-architectures#google-tpu) has another excellent animation of a systolic array multiplication for a fixed weight matrix.

### Appendix C: More on TPU internals

### Scalar Core

The scalar core is the control unit of the TPU. It fetches and dispatches all instructions and executes transfers from HBM into VMEM, and can be programmed to do scalar metadata work. Because the scalar core is single-threaded, one side-effect of this is that each core of the TPU is only capable of creating one DMA request per cycle.

To put this in context, a single scalar core controls a VPU consisting of 2048 ALUs, 4 MXUs, 2 XLUs, and multiple DMA engines. The highly skewed nature of control per unit compute is a source of hardware efficiency, but also limits the ability to do data dependent vectorization in any interesting way.

### VPU

The TPU vector core consists of a two dimensional SIMD vector machine (the **VPU**) that performs vector operations like vadd (vector addition) or vmax (elementwise max) and a set of vector registers called **VREGs** that hold data for the VPU and MXU. Each TPU core for v5p has 64 32-bit VREGs (32 in v4), giving us a total of about `64 * 8 * 128 * 4 = 256kB` of VREG memory.

The VPU is effectively a 2D vector arithmetic unit of shape `(8, 128)` where the 128 dimension is referred to as lane axis and the dimension of 8 is referred to as the sublane axis. Each (lane, sublane) pair on v5 contains 4 standard floating-point and integer ALUs. From a software point-of-view, this creates the appearance of a 8x128 vector unit with a total of 4048 floating point adders in v5.

The VPU executes most arithmetic instructions in one cycle in each of its ALUs (like vadd or vector add) with a latency of 2 cycles, so e.g. in v5 you can add 4 pairs of f32 values together from VREGs in each cycle. A typical VPU instruction might look like `{v2 = vadd.8x128.f32 v0, v1}` where v0 and v1 are input VREGs and v2 is an output VREG.

All lanes and sublanes execute the same program every cycle in a pure SIMD manner, but each ALU can perform a different operation. So we can e.g. process 1 vadd and 1 vsub in a single cycle, each of which operates on two full VREGs and writes the output to a third.

Reductions within a lane (over the size-8 sublane dimension) are cheap and very efficient (3 permutes and 3 adds). Cross-lane reductions are harder and involve the XLU or "cross lane unit", which is slow and fairly expensive.