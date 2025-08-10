---
layout: distill
title: "Sharded Matrices and How to Multiply Them"
# permalink: /main/
description: "여기서는 가장 큰 ML 모델들이 어떻게 여러 가속기에 걸쳐 분할(또는 '샤딩(sharded)')되는지 설명합니다. LLM은 대부분 행렬 곱셈으로 이루어져 있으므로, 이를 이해하는 것은 결국 행렬이 여러 디바이스에 분할되어 있을 때 어떻게 곱하는지를 이해하는 것으로 귀결됩니다. 저희는 TPU 통신 기본 연산(primitive)의 비용에 기반한 샤딩된 행렬 곱셈의 간단한 이론을 개발합니다."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 3

previous_section_url: "../tpus"
previous_section_name: "Part 2: TPUs"

next_section_url: ../transformers
next_section_name: "Part 4: Transformer Math"

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
  - name: "Partitioning Notation and Collective Operations"
  - subsections:
    - name: "A unified notation for sharding"
    - name: "How do we describe this in code?"
  - name: "Computation With Sharded Arrays"
  - subsections:
    - name: "Case 1: neither multiplicand has a sharded contracting dimension"
    - name: "Case 2: one multiplicand has a sharded contracting dimension"
    - name: "Case 3: both multiplicands have sharded contracting dimensions"
    - name: "Case 4: both multiplicands have a non-contracting dimension sharded along the same axis"
  - name: "A Deeper Dive into TPU Communication Primitives"
  - subsections:
    - name: "Our final communication primitive: the AllToAll"
    - name: "More about the ReduceScatter"
    - name: "How to overlap matmul communication with compute"
  - name: "What Have We Learned?"
  - name: "Some Problems to Work"

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

## Partitioning Notation and Collective Operations

수만 개의 TPU에서 LLM을 훈련할 때도, 추상적으로는 하나의 TPU에서 훈련할 때와 동일한 계산을 수행합니다. 차이점은 **우리의 배열이 단일 TPU의 HBM에 들어가지 않아서** 분할해야 한다는 것입니다.<d-footnote>속도를 위해 병렬화를 선택할 수도 있다는 점은 주목할 가치가 있습니다. 더 적은 수의 칩에 넣을 수 있더라도, 더 많은 칩으로 확장하면 단순히 더 많은 FLOPs/s를 얻을 수 있습니다. 예를 들어, 추론 중에는 더 작은 토폴로지에 맞출 수 있지만 지연 시간을 줄이기 위해 더 큰 토폴로지로 확장하기도 합니다. 마찬가지로, 훈련 중에는 스텝 시간을 줄이기 위해 더 많은 칩으로 확장하는 경우가 많습니다.</d-footnote> 우리는 이를 배열을 "*샤딩(sharding)*" 또는 "*파티셔닝(partitioning)*"한다고 말합니다.

다음은 4개의 TPU에 걸쳐 샤딩된 2D 배열 **A**의 예입니다:

{% include figure.liquid path="assets/img/sharding-example.png" class="img-fluid" caption="<b>Figure:</b><b>A</b>[I, J] 형태의 예제 배열이 4개의 디바이스에 걸쳐 샤딩됩니다. 두 차원 모두 <b>A</b>[I<sub>X</sub>, J<sub>Y</sub>] 샤딩으로 2개의 디바이스에 걸쳐 균등하게 샤딩됩니다. 각 TPU는 전체 메모리의 1/4을 보유합니다." %}

샤딩된 배열은 여전히 샤딩되지 않은 배열과 동일한 *전역(global)* 또는 *논리적 형태(logical shape)*(예: `(4, 128)`)를 가지지만, `(2, 64)`와 같은 *디바이스 로컬 형태(device local shape)*도 가집니다. 이는 각 TPU가 실제로 보유하고 있는 바이트 단위의 크기를 알려줍니다(위 그림에서 각 TPU는 전체 배열의 ¼을 보유함). 이제 이를 임의의 배열로 일반화해 보겠습니다.

### A unified notation for sharding

우리는 텐서가 디바이스에 걸쳐 블록으로 *어떻게* 샤딩되는지를 설명하기 위해 *named-axis notation*의 변형을 사용합니다: **X**, **Y, Z**와 같이 **메시 축 이름(mesh axis names)**이 부여된 2D 또는 3D 디바이스 그리드인 **디바이스 메시(device mesh)**가 있다고 가정합니다. 그런 다음 배열의 각 이름 있는 차원이 물리적 메시 축에 걸쳐 어떻게 분할되는지를 설명함으로써 행렬 데이터가 디바이스 메시에 어떻게 배치되는지를 지정할 수 있습니다. 우리는 이 할당을 **샤딩(sharding)**이라고 부릅니다.

**예제 (위 다이어그램)**: 위 다이어그램의 경우, 다음과 같습니다:
* **Sharding:** $A[I_X, J_Y]$, 이는 첫 번째 축 $I$를 메시 축 $X$를 따라 샤딩하고, 두 번째 축 $J$를 메시 축 $Y$를 따라 샤딩하라는 것을 알려줍니다. 이 샤딩은 각 샤드가 배열의 $1 / (\lvert X\rvert \cdot \lvert Y\rvert)$ 를 보유함을 알려줍니다.
* **Mesh:** 위 디바이스 메시 `Mesh(devices=((0, 1), (2, 3)), axis_names=(‘X', ‘Y'))` 는 4개의 TPU가 2x2 그리드에 있으며, 축 이름이 $X$와 $Y$임을 알려줍니다.

이 둘을 종합하면, 배열의 로컬 형태(개별 디바이스가 보유하는 샤드의 크기)는 $(\lvert I\rvert / 2, \lvert J\rvert / 2)$ 임을 알 수 있습니다. 여기서 $$\lvert I\rvert$$ 는 A의 첫 번째 차원 크기이고 $$\lvert J\rvert$$ 는 A의 두 번째 차원 크기입니다.

**Example (1개 축에 대한 2D 샤딩)**: $A[I_{XY}, J]$ 는 첫 번째 차원(I)을 X와 Y 하드웨어 축 모두에 걸쳐 샤딩합니다. 디바이스당 바이트 수는 이전 샤딩과 동일하지만 로컬 형태는 다릅니다. 이제 $(\lvert I\rvert /(\lvert X\rvert \cdot \lvert Y\rvert), \lvert J\rvert)$ 입니다.

**Visualizing these shardings:** 4개의 디바이스에 분할된 2D 데이터 배열을 보며 이러한 샤딩을 시각화해 봅시다:

{% include figure.liquid path="assets/img/sharding-colored1.png" class="img-fluid img-small" %}

행렬의 *완전 복제(fully-replicated)* 형태는 샤딩 할당 없이 단순히 $A[I, J]$로 씁니다. 이는 *각* 디바이스가 전체 행렬의 완전한 복사본을 포함함을 의미합니다.

{% include figure.liquid path="assets/img/sharding-colored2.png" class="img-fluid img-small" %}

이러한 차원 중 하나가 메시 축에 걸쳐 분할되었음을 나타내고 싶을 때는 메시 축 아래 첨자를 사용합니다. 예를 들어 $A[I_X, J]$는 **I** 논리적 축이 **X** 메시 차원에 걸쳐 분할되었지만, **J** 차원은 분할되지 *않았으며*, 블록이 **Y** 메시 축에 걸쳐 *부분적으로 복제(partially-replicated)*되어 있음을 의미합니다.

{% include figure.liquid path="assets/img/sharding-colored3.png" class="img-fluid img-small" %}

$A[I_X, J_Y]$ 는 **I** 논리적 축이 **X** 메시 축에 걸쳐 분할되었고, **J** 차원이 **Y** 메시 축에 걸쳐 분할되었음을 의미합니다.

{% include figure.liquid path="assets/img/sharding-colored4.png" class="img-fluid img-small" %}

아래 그림에서 다른 가능성들을 보여줍니다:

{% include figure.liquid path="assets/img/sharding-colored5.png" class="img-fluid" %}

여기서 $A[I_{XY}, J]$ 는 **X**와 **Y** 메시 축을 더 큰 평탄화된 차원으로 취급하고, **I** 이름 있는 축을 모든 디바이스에 걸쳐 분할함을 의미합니다. 여러 메시 축 아래 첨자의 순서는 그리드에 걸친 분할(partitioning)의 순회 순서(traversal order)를 지정하므로 중요합니다.

{% include figure.liquid path="assets/img/sharding-colored6.png" class="img-fluid img-small" %}

마지막으로, 여러 이름 있는 축이 *동일한* 메시 차원을 따라 샤딩될 수 *없다*는 점에 유의하세요. 예를 들어 $A[I_X, J_X]$는 의미가 없는 금지된 샤딩입니다. 메시 차원이 배열의 한 차원을 샤딩하는 데 사용되면, 그것은 일종의 "소진된" 상태가 됩니다.

<b markdown=1 style="color: #57cf57;">Pop Quiz:</b> **A**가 `int8[128, 2048]` 형태의 배열이고, 샤딩이 $A[I_{XY}, J]$이며, 메시가 `Mesh({‘X': 2, ‘Y': 8, ‘Z': 2})`(총 32개 디바이스)라고 가정한다면, **A**는 디바이스당 얼마나 많은 메모리를 사용할까요? 모든 디바이스에 걸쳐 총 얼마나 많은 메모리를 사용할까요?

{% details 답을 보려면 여기를 클릭하세요. %}

**Answer:** 배열 **A**는 X와 Y에 걸쳐 샤딩되고 Z에 걸쳐 복제되므로, 디바이스당 형태는 `int8[128 / (2 * 8), 2048] = int8[8, 2048]`이며, 크기는 `8 * 2048 = 16,384` 바이트입니다. Z에 걸쳐 복제되므로, X와 Y에 걸쳐 완전히 샤딩된 Z-평면마다 복사본이 하나씩 있고, 그런 평면이 2개 있으므로, 총 크기(모든 디바이스에 걸쳐)는 `128 * 2048 * 2 = 512kiB`입니다.

{% enddetails %}

### How do we describe this in code?

JAX는 위에서 설명한 추상적인 구문과 매우 유사한 이름 있는 샤딩 구문을 사용합니다. 이에 대해서는 [섹션 10](../jax-stuff)에서 더 자세히 이야기하겠지만, 간단히 미리 살펴보겠습니다. You can play with this in a Google Colab [여기](https://colab.research.google.com/drive/15cxw66eABwZPG-V4QFmbLfiykPFf_gaP?usp=sharing) 에서 이것을 직접 실행해보고 결과를 프로파일링하여 JAX가 다른 샤딩을 어떻게 처리하는지 볼 수 있습니다. 이 스니펫은 3가지 작업을 수행합니다:

1. 8개의 TPU를 'X'와 'Y'라는 이름이 할당된 두 축을 가진 4x2 그리드로 매핑하는 **jax.Mesh**를 생성합니다. 
2. 행렬 A와 B를 생성하는데, A는 두 차원 모두에 걸쳐 샤딩되고 B는 출력 차원을 따라 샤딩됩니다.
3. 샤딩된 배열을 반환하는 간단한 행렬 곱셈을 컴파일하고 수행합니다.

```py
import jax
import jax.numpy as jnp

# Create our mesh! We're running on a TPU v2-8 4x2 slice with names 'X' and 'Y'.
assert len(jax.devices()) == 8
mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=('X', 'Y'))

# A little utility function to help define our sharding. A PartitionSpec is our
# sharding (a mapping from axes to names).
def P(*args):
  return jax.NamedSharding(mesh, jax.sharding.PartitionSpec(*args))

# We shard both A and B over the non-contracting dimension and A over the contracting dim.
A = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=P('X', 'Y'))
B = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=P(None, 'Y'))

# We can perform a matmul on these sharded arrays! out_shardings tells us how we want
# the output to be sharded. JAX/XLA handles the rest of the sharding for us.
y = jax.jit(lambda A, B: jnp.einsum('BD,DF->BF', A, B), out_shardings=P('X', 'Y'))(A, B)
```

JAX의 멋진 점은 이 배열들이 샤딩되지 않은 것처럼 동작한다는 것입니다! `B.shape`는 전역 또는 논리적 형태(2048, 8192)를 알려줄 것입니다. 로컬로 어떻게 샤딩되었는지 보려면 `B.addressable_shards`를 실제로 봐야 합니다. 이 배열들에 대해 연산을 수행하면 JAX는 연산을 수행하기 위해 어떻게 브로드캐스트하거나 재구성할지 알아서 처리합니다. 예를 들어, 위 예에서 **A**의 로컬 형태는 `[2, 1024]`이고 **B**의 로컬 형태는 `[2048, 4096]`입니다. JAX/XLA는 최종 곱셈을 수행하기 위해 필요에 따라 이 배열들 간의 통신을 자동으로 추가합니다.

## Computation With Sharded Arrays

여러 디바이스에 분산된 데이터 배열이 있고 이에 대해 수학적 연산을 수행하고자 할 때, 데이터와 계산을 모두 샤딩하는 데 관련된 오버헤드는 무엇일까요?

당연히, 이것은 관련된 계산에 따라 다릅니다.

* *원소별(elementwise)* 연산의 경우, 분산된 배열에 대해 연산하는 데 **오버헤드가 없습니다**.
* 여러 디바이스에 상주하는 원소들에 걸쳐 연산을 수행하고자 할 때, 상황은 복잡해집니다. 다행히도, 대부분의 머신러닝에서 거의 모든 계산은 행렬 곱셈 형태로 이루어지며, 이는 비교적 간단하게 분석할 수 있습니다.

이 섹션의 나머지 부분에서는 샤딩된 행렬을 곱하는 방법을 다룹니다. 대략적으로 말해, 이것은 각 청크를 완전히 곱하거나 합산할 수 있도록 행렬의 청크를 이동시키는 것을 포함합니다. **각 샤딩은 다른 통신을 포함합니다.** 예를 들어, $A[I_X, J] \cdot B[J, K_Y] \to C[I_X, K_Y]$는 *축소 차원*(실제로 합산하는 차원인 J)이 샤딩되지 않았기 때문에 통신 없이 곱셈이 가능합니다. 하지만 출력이 샤딩되지 않기를 원한다면(즉, $A[I_X, J] \cdot B[J, K_Y] \to C[I, K]$), $A$ 또는 $C$를 모든 디바이스에 복사해야 합니다. 이 두 가지 선택은 다른 통신 비용을 가지므로, 이 비용을 계산하고 가장 낮은 것을 선택해야 합니다.

{% details 이것을 "블록 행렬 곱셈"으로 생각할 수 있습니다. %}

먼저 "블록 행렬", 즉 행렬의 중첩된 행렬 개념을 상기해 봅시다:

$$\begin{equation}
\begin{pmatrix}
a_{00} & a_{01} & a_{02} & a_{03} \\
a_{10} & a_{11} & a_{12} & a_{13} \\
a_{20} & a_{21} & a_{22} & a_{23} \\
a_{30} & a_{31} & a_{32} & a_{33}
\end{pmatrix}
=
\left(
\begin{matrix}
\begin{bmatrix}
a_{00} & a_{01} \\
a_{10} & a_{11}
\end{bmatrix} \\
\begin{bmatrix}
a_{20} & a_{21} \\
a_{30} & a_{31}
\end{bmatrix}
\end{matrix}
\begin{matrix}
\begin{bmatrix}
a_{02} & a_{03} \\
a_{12} & a_{13}
\end{bmatrix} \\
\begin{bmatrix}
a_{22} & a_{23} \\
a_{32} & a_{33}
\end{bmatrix}
\end{matrix}
\right)
=
\begin{pmatrix}
\mathbf{A_{00}} & \mathbf{A_{01}} \\
\mathbf{A_{10}} & \mathbf{A_{11}}
\end{pmatrix}
\end{equation}$$

행렬 곱셈은 피곱셈 행렬이 블록으로 작성될 때, 곱이 표준 규칙에 따라 블록 행렬 곱셈으로 작성될 수 있다는 좋은 속성을 가집니다:

$$\begin{equation}
\begin{pmatrix}
A_{00} & A_{01} \\
A_{10} & A_{11}
\end{pmatrix}
\cdot
\begin{pmatrix}
B_{00} & B_{01} \\
B_{10} & B_{11}
\end{pmatrix}
=
\begin{pmatrix}
A_{00}B_{00} + A_{01}B_{10} & A_{00}B_{01} + A_{01}B_{11} \\
A_{10}B_{00} + A_{11}B_{10} & A_{10}B_{01} + A_{11}B_{11}
\end{pmatrix}
\end{equation}$$

이것이 의미하는 바는, 분산 행렬 곱셈을 구현하는 것은 이 샤딩된 블록들을 네트워크를 통해 이동시키고, 블록에 대해 *로컬* 행렬 곱셈을 수행하고, 그 결과를 합산하는 것으로 귀결된다는 것입니다. **문제는 어떤 통신을 추가하고, 그것이 얼마나 비싼가입니다.**

{% enddetails %}

편리하게도, 모든 가능한 샤딩을 우리가 고려해야 할 대략 4가지 경우로 요약할 수 있으며, 각각은 어떤 통신을 추가해야 하는지에 대한 규칙을 가집니다.
1. **[Case 1](#case-1-neither-multiplicand-has-a-sharded-contracting-dimension):** 어느 피곱셈 행렬도 축소 차원이 샤딩되지 않음. _통신 없이 로컬 샤드를 곱할 수 있습니다._
2. **[Case 2](#case-2-one-multiplicand-has-a-sharded-contracting-dimension):** 한 피곱셈 행렬에 샤딩된 축소 차원이 있음. _일반적으로 축소 차원을 따라 샤딩된 입력을 "AllGather"합니다._
3. **[Case 3](#case-3-both-multiplicands-have-sharded-contracting-dimensions):** 두 피곱셈 행렬 모두에 샤딩된 축소 차원이 있음. _로컬 샤드를 곱한 다음, 결과를 "AllReduce"할 수 있습니다._
4. **[Case 4](#case-4-both-multiplicands-have-a-non-contracting-dimension-sharded-along-the-same-axis):** 두 피곱셈 행렬 모두 동일한 축을 따라 샤딩된 비축소 차원을 가짐. 두 입력 중 하나를 먼저 AllGather하지 않고는 진행할 수 없습니다.

이것들을 단순히 따라야 할 규칙으로 생각할 수도 있지만, 이 규칙들이 왜 성립하고 얼마나 비싼지를 이해하는 것도 가치가 있습니다. 이제 각각을 자세히 살펴보겠습니다.

### Case 1: 어느 피곱셈 행렬도 샤딩된 축소 차원을 가지지 않음

**Lemma:** 분할된 텐서를 곱할 때, 계산은 유효하며 출력은 입력의 샤딩을 따릅니다. *단, 축소 차원이 샤딩되거나 두 텐서 모두 동일한 축을 따라 샤딩된 비축소 차원을 가지는 경우는 예외입니다.* 예를 들어, 다음은 잘 작동합니다.

$$\begin{equation*}
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K_Y] \rightarrow \mathbf{C}[I_X, K_Y]
\end{equation*}$$

전혀 통신 없이, 그리고 결과는 X와 Y 하드웨어 차원 모두에 걸쳐 샤딩된 텐서가 됩니다. 왜 그런지 생각해보세요. 기본적으로, 계산은 샤딩과 *독립적*입니다. 왜냐하면 각 배치 항목은 곱하고 축소할 수 있는 축소 중인 축의 로컬 청크를 가지고 있기 때문입니다. 다음 사례들 중 어떤 것이든 잘 작동하며 이 규칙을 따릅니다:

$$\begin{align*}
\mathbf{A}[I, J] \cdot \mathbf{B}[J, K] \rightarrow &\ \mathbf{C}[I, K] \\
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K] \rightarrow &\ \mathbf{C}[I_X, K]\\
\mathbf{A}[I, J] \cdot \mathbf{B}[J, K_Y] \rightarrow &\ \mathbf{C}[I, K_Y]\\
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K_Y] \rightarrow &\ \mathbf{C}[I_X, K_Y]
\end{align*}$$

**A**나 **B** 모두 샤딩된 축소 차원 **J**를 가지지 않으므로, 입력의 로컬 블록 행렬 곱셈을 수행하기만 하면 결과는 *이미* 원하는 출력 샤딩에 따라 샤딩되어 있습니다. 두 피곱셈 행렬 모두 동일한 축을 따라 샤딩된 비축소 차원을 가질 때는 이것이 더 이상 사실이 아닙니다([invalid shardings(유효하지 않은 샤딩)](#case-4-both-multiplicands-have-a-non-contracting-dimension-sharded-along-the-same-axis) 섹션 참조).

### Case 2: 한 피곱셈 행렬이 샤딩된 축소 차원을 가짐

축소 **J** 차원에서 샤딩된 **A**와 완전 복제된 **B**의 분산 행렬 곱셈의 간단한 경우를 고려해 봅시다:

$$\mathbf{A}[I, J_X] \cdot \mathbf{B}[J, K] \rightarrow \mathbf{C}[I, K]$$

로컬 **A**, **B** 블록을 서로 곱하는 로컬 행렬 곱셈을 단순히 수행할 수 없습니다. **A**의 축소 축에서 전체 데이터가 없기 때문입니다. 일반적으로, 먼저 **A**의 샤드를 로컬에서 "**AllGather**"한 다음, **B**와 곱합니다:

$$\textbf{AllGather}_X[I, J_X] \rightarrow \mathbf{A}[I, J]$$

$$\mathbf{A}[I, J] \cdot \mathbf{B}[J, K] \rightarrow \mathbf{C}[I, K]$$

AllGather는 축을 따라 *샤딩을 제거*하고 디바이스에 걸쳐 퍼져 있는 샤드를 해당 축을 따라 *각* 디바이스에 재조립합니다. 위 표기법을 사용하면, AllGather는 축 집합에서 아래 첨자를 제거합니다. 예:

$$\textbf{AllGather}_{XY}(A[I_{XY}, J]) \rightarrow A[I, J]$$

주어진 차원에 대해 모든 아래 첨자를 제거할 필요는 없습니다. 예: $$A[I_{XY}, J] \rightarrow A[I_Y, J]$$ 도 단일 축에 대한 AllGather입니다.

*non-contracting(비축소)* 차원 샤딩을 제거하기 위해 AllGather를 사용할 수도 있습니다. 예를 들어, 행렬 곱셈:

$$A[I_X, J] \cdot B[J, K] \rightarrow C[I, K]$$

We could either AllGather **A** initially to remove the input sharding, or we can do the sharded matmul and then AllGather the result **C**.

**How is an AllGather actually performed?** To perform a 1-dimensional AllGather around a single TPU axis (a ring), we basically have each TPU pass its shard around a ring until every device has a copy.<d-footnote>A GPU AllGather can also work like this, where you create a ring out of the GPUs in a node and pass the chunks around in that (arbitrary) order.</d-footnote> Here is an animation:

출력 샤딩을 제거하기 위해 유사하게 **X**를 따라 AllGather를 수행할 수 있습니다. 하지만 축소 차원을 AllGather하는 경우와는 달리, 이 경우에는 행렬 곱셈을 수행하기 전이나 후에 AllGather를 할 수 있습니다. 축소 차원은 반드시 행렬 곱셈을 수행하기 전에 AllGather해야 합니다.

**AllGather는 실제로 어떻게 수행될까요?** 단일 축을 따라 AllGather를 수행하려면, 모든 디바이스가 복사본을 가질 때까지 모든 샤드를 축 주위로 전달해야 합니다. Figure 1은 한 예시를 보여줍니다. 8개의 각 디바이스는 배열의 1/8로 시작하여 모든 복사본으로 끝납니다. 이를 효율적으로 수행하는 한 가지 방법은 각 디바이스가 sharding dimension ring 주위로 자신의 샤드를 한 방향 또는 양방향으로 전달하는 것입니다. 한 방향으로 하면 링크당 $$\text{total size} / N$$ 크기의 홉이 $$N - 1$$번 필요하고, 양방향으로 하면 링크당 $$2 \cdot \text{total size} / N$$ 크기의 홉이 $\lceil \frac{N}{2} \rceil$ 번 필요합니다.

We can either do an AllGather in one direction or both directions (two directions is shown above). If we do one direction, each TPU sends chunks of size $\text{bytes} / N$ over $N - 1$ hops around the ring. If we do two directions, we have $\lfoor \frac{N}{2} \rfloor$ hops of size $2 \cdot \text{bytes} / N$.

**이것은 얼마나 오래 걸릴까요?** 양방향 AllGather를 예로 들어 얼마나 오래 걸리는지 계산해 봅시다. $$V$$를 배열의 바이트 수, $$\lvert X\rvert$$를 축소 차원의 샤드 수라고 합시다. 위 다이어그램에서 각 홉은 각 방향으로 $V / \lvert X\rvert$ 바이트를 보내므로, 각 홉은 다음과 같은 시간이 걸립니다.

$$T_{hop} = \frac{2 \cdot V}{X \cdot W_\text{ici}}$$

여기서 $$W_\text{ici}$$는 **양방향(bidirectional)** ICI 대역폭입니다.<d-footnote>분자의 2는 양방향 대역폭을 사용하고 있다는 사실에서 비롯됩니다. 각 방향으로 $V / |X|$를 보내므로 총 $2V / |X|$를 보냅니다.</d-footnote> 모든 TPU에 도달하기 위해 총 $\lvert X\rvert / 2$ 홉을 보내야 하므로<d-footnote>기술적으로는 $\lceil | X | / 2 \rceil$</d-footnote>, 전체 축소에는 다음과 같은 시간이 걸립니다.

$$T_{total} = \frac{2 \cdot V \cdot X}{2 \cdot X \cdot W_\text{ici}}$$

$$T_{total} = \frac{V}{W_\text{ici}}$$

이것이 **$$\lvert X\rvert$$에 의존하지 않는다는 점에 유의하세요!** 이는 꽤 놀라운 사실인데, 왜냐하면 우리 TPU가 로컬로만 연결되어 있음에도 불구하고 연결의 지역성이 중요하지 않다는 것을 의미하기 때문입니다. 우리는 단지 각 링크의 속도에 의해 병목 현상을 겪습니다.

<p markdown=1 class="takeaway">**Takeaway:** throughput-bound 환경에서 AllGather(또는 ReduceScatter나 AllReduce)를 수행할 때, 실제 통신 시간은 배열이 샤딩된 디바이스의 수가 아니라 배열의 크기와 사용 가능한 대역폭에만 의존합니다!</p>

**ICI 지연 시간에 대한 참고 사항:** ICI 링크를 통한 각 홉은 데이터 양과 관계없이 고유한 오버헤드를 가집니다. 이는 보통 약 1us입니다. 이는 배열 $$A$$가 매우 작고 각 홉이 1us 미만으로 걸릴 때, 계산이 $$\lvert X \rvert$$에 _의존하는_ "지연 시간 제한적인(latency-bound)" 환경에 들어갈 수 있음을 의미합니다.

{% details 자세한 내용은 여기를 클릭하세요. %}

$$T_\text{min}$$ 을 단일 홉의 최소 시간이라고 합시다. 이 경우에

$$T_{hop} = \max \left[ T_{min}, \frac{2 \cdot V}{X \cdot W_\text{ici}} \right]$$

$$T_{total} = \max \left[ \frac{T_{min} \cdot X}{2}, \frac{V}{W_\text{ici}} \right]$$

이 됩니다. 왜냐하면 우리는 $$\lvert X \rvert / 2$$ 홉을 수행하기 때문입니다. 큰 축소나 수집의 경우, 우리는 확실히 대역폭 제한적입니다. 너무 많은 데이터를 보내고 있어서 각 홉의 오버헤드는 거의 무시할 수 있습니다. 하지만 작은 배열(예: 모델에서 샘플링할 때)의 경우, 이는 무시할 수 없으며 ICI 대역폭은 관련이 없습니다. 우리는 순수하게 지연 시간에 의해 제한됩니다. 다른 말로 하면, 특정 TPU(예: `4.5e10` 단방향 ICI 대역폭을 가진 TPU v5e)가 주어졌을 때, `4.5e10 * 1e-6 = 45kB` 미만의 버퍼를 보내는 것은 지연 시간 제한적이 될 것입니다.

{% enddetails %}

다음은 TPU v5e 8x16 슬라이스에서 AllGather 대역폭을 실증적으로 측정한 것입니다. 배열은 16개 축에 걸쳐 샤딩되어 완전한 양방향 링을 가집니다.

{% include figure.liquid path="assets/img/all-gather-bandwidth.png" class="img-small" caption="<b>Figure:</b> AllGather 중 TPU v5e의 실증적 대역폭 및 추정 링크 대역폭. 주황색 BW는 AllGather된 실제 초당 바이트 수이며, 파란색 곡선은 집합 연산의 알려진 비용에 따라 계산된 실증적 단방향 링크 대역폭을 보여줍니다." %}

주장된 최대 대역폭(`4.5e10`)의 약 95%만 달성하고, 이 최대치를 약 10MB에서만 달성한다는 점에 유의하세요. 16방향으로 샤딩되면 디바이스당 약 500kB가 됩니다.

**여러 축에 걸쳐 AllGather를 수행하면 어떻게 될까요?** 여러 축에 걸쳐 수집할 때, 우리는 수집을 수행할 여러 차원의 ICI를 가집니다. 예를 들어, AllGather<sub>XY</sub>([B, D<sub>XY</sub>])는 두 개의 하드웨어 메시 축에 걸쳐 작동합니다. 이는 사용 가능한 대역폭을 $$n_\text{axes}$$ 배만큼 증가시킵니다.

{% details 자세한 내용은 여기를 클릭하세요. %}

일반적으로 우리는

$$T_{total} = \max \left[ \frac{T_{min} \cdot \sum_{i} |X_i|}{2}, \frac{V}{W_\text{ici} \cdot n_\text{axes}} \right]$$

를 가집니다. 여기서 $$\sum_i \lvert X_i \rvert / 2$$는 TPU 메시에서 가장 긴 경로의 길이입니다.

{% enddetails %}

<b markdown=1 style="color:rgb(144, 92, 255);">Pop Quiz 2 [AllGather time]:</b> [파트 2](../tpus)의 수치를 사용하여, 2D 메시 `{'X': 8, 'Y': 4}`를 가진 TPUv5e에서 $$E = 2048$$, $$F = 8192$$인 bfloat16의 AllGather<sub>Y</sub>([E<sub>Y</sub>, F]) → [E, F]를 수행하는 데 얼마나 걸릴까요? $$E=256, F=256$$일 때는 어떨까요?

{% details 답을 보려면 여기를 클릭하세요. %}

**Answer:** 몇 가지 기본 수를 계산하는 것으로 시작하겠습니다:

1) TPU v5e는 2개의 각 축에 대해 초당 4.5e10 바이트의 단방향 ICI 대역폭을 가집니다.
2) (a)의 bfloat16에서, 우리는 $A[E_Y, F]$를 가지므로 각 디바이스는 bfloat16[512, 8192] 형태의 배열을 보유하며, 이는 512 * 8192 * 2 = 8.4MB의 크기를 가집니다. 전체 배열의 크기는 2048 * 8192 * 2 = 34MB입니다.

*파트 (1)의 경우*, 위 공식을 사용할 수 있습니다. 한 축에 대해서만 AllGather를 수행하므로, $T_{\text{comms}} = 34e6 / 9e10 = 377\mu s$입니다. latency bound인지 확인하기 위해, 크기 4의 축에서는 최대 3개의 홉이 있으므로 지연 시간 한계는 약 3us 정도이므로 근접하지 않습니다. 하지만 TPU v5e는 한 축의 크기가 16일 때만 랩어라운드 연결을 가지므로, 여기서는 *실제로 완전한 양방향 AllGather를 할 수 없습니다*. 가장자리에서 다른 가장자리로 데이터가 도달하려면 3개의 홉이 필요하므로, 이론적으로는 $T_{\text{comms}} = 3 * 8.4e6 / 4.5e10 = 560\mu s$에 가깝습니다. [이 Colab](https://colab.research.google.com/drive/15tDZMfNqm2vJjvSzw5VC9qtSwc5td-oV?usp=sharing)의 [**실제 프로파일**](https://imgur.com/a/RkvpRGQ) 은 $680 \mu s$를 보여주는데, 이는 이론적 대역폭의 100%를 얻지 못할 가능성이 높기 때문에 합리적입니다! *파트 (2)의 경우* 각 샤드의 크기는 `64 * 256 * 2 = 32kB`입니다. `32e3 / 4.5e10 = 0.7us`이므로, 우리는 latency bound입니다. 3개의 홉이 있으므로, 대략 3 * 1us = 3us가 걸릴 것입니다. [실제로는 8us에 가깝습니다.](https://imgur.com/a/HZLQmYs)

{% enddetails %}

### Case 3: 두 피연산자 모두 축소 차원이 분할된 경우

세 번째 기본 경우는 두 피연산자 모두 동일한 메시 축을 따라 축소 차원에 샤딩된 경우입니다:

$$\textbf{A}[I, J_X] \cdot \textbf{B}[J_X, K] \rightarrow C[I, K]$$

이 경우, *로컬* 분할된 블록 행렬 곱셈은 동일한 축소 인덱스 집합을 공유하므로 최소한 수행이 *가능*합니다. 하지만 각 곱은 원하는 전체 곱의 *부분 합*만을 나타낼 것이며, **X** 차원을 따르는 각 디바이스는 이 최종 원하는 곱의 다른 *부분 합*을 가지게 될 것입니다. 이는 매우 흔한 경우이므로, 이 조건을 명시적으로 표시하기 위해 표기법을 확장합니다:

$$\textbf{A}[I, J_X] \cdot_\text{LOCAL} \textbf{B}[J_X, K] \rightarrow C[I, K] \{\ U_X \}$$

**{ U\<sub\>X\</sub\> }** 표기법은 "X 메시 축에 대해 **축소되지 않음(unreduced)**"으로 읽으며, 연산이 최종 합산이 보류된 채 "미완성" 상태임을 의미합니다. $\cdot_\text{LOCAL}$ 구문은 로컬 합계를 수행하지만 결과를 축소되지 않은 상태로 남겨둔다는 것을 의미합니다.

이는 행렬 곱셈과 외적(outer product)에 대한 다음 결과로 볼 수 있습니다:

$$A \cdot B = \sum_{i=1}^{P} \underbrace{A_{:,i} \otimes B_{i,:}}_{\in \mathbb{R}^{n \times m}}$$

where ⊗ is the outer product. Thus, if TPU **i** on axis **X** has the **i**th column of **A**, and the **i**th row of **B**, we can do a local matrix multiplication to obtain $$A_{:,i} \otimes B_{i,:} \in \mathbb{R}_{n\times m}$$. This matrix has, in each entry, the **i**th term of the sum that **A • B** has at that entry. We still need to perform that sum over **P**, which we sharded over mesh axis **X**, to obtain the full **A • B**. This works the same way if we write **A** and **B** by blocks (i.e. shards), and then sum over each resulting shard of the result.

We can perform this summation using a full **AllReduce** across the **X** axis to remedy this:

$$\begin{align*}
A[I, J_X] \cdot_\text{LOCAL} B[J_X, K] \rightarrow &\ C[I, K] \{ U_X \} \\
\textbf{AllReduce}_X C[I, K] \{ U_X \} \rightarrow &\ C[I, K]
\end{align*}$$

AllReduce removes partial sums, resulting in *each* device along the axis having the same fully-summed value. AllReduce is the second of several key communications we'll discuss in this section, the first being the AllGather, and the others being ReduceScatter and AllToAll. An AllReduce takes an array with an unreduced (partially summed) axis and performs the sum by passing those shards around the unreduced axis and accumulating the result. The signature is

$$\textbf{AllReduce}_Y A[I_X, J] \{U_Y\} \rightarrow A[I_X, J]$$

This means it simply removes the $\\{U_Y\\}$ suffix but otherwise leaves the result unchanged.

**How expensive is an AllReduce?** One mental model for how an AllReduce is performed is that every device sends its shard to its neighbors, and sums up all the shards that it receives. Clearly, this is more expensive than an AllGather because each "shard" has the same shape as the full array. Generally, **an AllReduce is twice as expensive as an AllGather.** One way to see this is to note that an **AllReduce** can be expressed as a composition of two other primitives: a **ReduceScatter** and an **AllGather**. Like an AllReduce, a ReduceScatter resolves partial sums on an array but results in an output 'scattered' or partitioned along a given dimension. AllGather collects all those pieces and 'unpartitions/unshards/replicates' the logical axis along that physical axis.

$$\begin{align*}
\textbf{ReduceScatter}_{Y,J} : A[I_X,J] \{U_Y\} \rightarrow &\ A[I_X, J_Y] \\
\textbf{AllGather}_Y : A[I_X, J_Y] \rightarrow &\ A[I_X, J]
\end{align*}$$

**What about a ReduceScatter?** Just as the AllReduce removes a subscript ($F_Y \to F$ above), a ReduceScatter sums an unreduced/partially summed array and then scatters (shards) a different logical axis along the same mesh axis. $[F]\\{U_Y\\} \to [F_Y]$. The animation shows how this is done: note that it's very similar to an AllGather but instead of retaining each shard, we sum them together. Thus, its latency is roughly the same, excluding the time taken to perform the reduction.

{% include figure.liquid path="assets/img/reduce-scatter.gif" class="img-fluid" %}

The communication time for each hop is simply the per-shard bytes $V / Y$ divided by the bandwidth $W_\text{ici}$, as it was for an AllGather, so we have

$$T_{\text{comms per AllGather or ReduceScatter}} = \frac{V}{W_\text{ici}}$$

$$T_{\text{comms per AllReduce}} = 2 \cdot \frac{V}{W_\text{ici}}$$

where $$W_\text{ici}$$ is the bidirectional bandwidth, so long as we have a full ring to reduce over.

### Case 4: both multiplicands have a non-contracting dimension sharded along the same axis

Each mesh dimension can appear at most once when sharding a tensor. Performing the above rules can sometimes lead to a situation where this rule is violated, such as:

$$A[I_X, J] \cdot B[J, K_X] \rightarrow C[I_X, K_X]$$

This is invalid because a given shard, say **i**, along dimension **X**, would have the **(i, i)**th shard of **C**, that is, a diagonal entry. There is not enough information among all shards, then, to recover anything but the diagonal entries of the result, so we cannot allow this sharding.

The way to resolve this is to AllGather some of the dimensions. Here we have two choices:

$$\begin{align*}
\textbf{AllGather}_X A[I_X, J] \rightarrow &\ A[I, J] \\
A[I, J] \cdot B[J, K_X] \rightarrow &\ C[I, K_X]
\end{align*}$$

or

$$\begin{align*}
\textbf{AllGather}_X B[J, K_X] \rightarrow &\ B[J, K] \\
A[I_X, J] \cdot B[J, K] \rightarrow &\ C[I_X, K]
\end{align*}$$

In either case, the result will only mention **X** once in its shape. Which one we pick will be based on what sharding the following operations need.

## A Deeper Dive into TPU Communication Primitives

The previous 4 cases have introduced several "core communication primitives" used to perform sharded matrix multiplications:

1. **AllGather:** removes a subscript from a sharding, gathering the shards.
2. **ReduceScatter:** removes an "un-reduced" suffix from an array by summing shards over that axis, leaving the array sharded over a second axis.
3. **AllReduce:** removes an "un-reduced" suffix, leaving the array unsharded along that axis.

There's one more core communication primitive to mention that arises in the case of Mixture of Experts (MoE) models and other computations: the **AllToAll**.

### Our final communication primitive: the AllToAll

A final fundamental collective which does not occur naturally when considering sharded matrix multiplies, but which comes up constantly in practice, is the **AllToAll** collective, or more precisely the special case of a *sharded transposition* or resharding operation. e.g.

$$\textbf{AllToAll}_{X, J} A[I_X, J] \rightarrow A[I, J_X]$$

AllToAlls are typically required to rearrange sharded layouts between different regions of a sharded computation that don't have compatible layout schemes. They arise naturally when considering sharded mixture-of-experts models. *You can think of an AllToAll as moving a subscript from one axis to another*. Because an all to all doesn't need to replicate all of the data of each shard across the ring, it's actually *cheaper* than an AllGather (by a factor of ¼)<d-footnote>For even-sized bidirectional rings, each device will send $(N/2 + (N/2-1) + … + 1)$ chunks right and $((N/2-1) + … + 1)$ chunks left $= 0.5 \cdot (N / 2) \cdot (N/2 + 1) + 0.5 \cdot (N / 2) \cdot (N/2 - 1) = N^2/4$. The size of each chunk (aka shard of a shard) is $\text{bytes} / N^2$ so the per-device cost is $(\text{bytes} / N^2) \cdot N^2 / 4 = \text{bytes} / 4$. This result scales across all devices as the total bandwidth scales with device number.</d-footnote>.

{% include figure.liquid path="assets/img/all-to-all.gif" class="img-fluid" %}

If we generalize to an ND AllToAll, the overall cost for an array of $V$ bytes on an AxBxC mesh is

$$T_\text{comms per AllToAll} = \frac{V \cdot \max(A, B, C, ...)}{4 \cdot N \cdot W_\text{ici}}$$

where as usual $W_\text{ici}$ is the bidirectional ICI bandwidth. For a 1D mesh, this reduces to $V / (4 \cdot W_\text{ici})$, which is 1 / 4 the cost of an AllReduce. In 2D, the cost actually scales down with the size of the smallest axis.

*Aside: If you want a hand-wavy derivation of this fact, start with a 1D torus $\mathbb{Z} / N\mathbb{Z}$. If we pick a source and target node at random, they are on average N / 4 hops from each other, giving us a cost of $(V \cdot N) / (4 * N)$. Now if we consider an ND torus, each axis is basically independent. Each node has $1 / N$ bytes and on average has to hop its data $\max(A, B, C, …) / 4$ hops.*

### More about the ReduceScatter

ReduceScatter is a more fundamental operation than it first appears, as it is actually the derivative of an AllGather, and vice versa. i.e. if in the forward pass we have:

$$\textbf{AllGather}_X A[I_X] \rightarrow A[I]$$

Then we ReduceScatter the reverse-mode derivatives **A'** (which will in general be different on each shard) to derive the sharded **A'**:

$$\textbf{ReduceScatter}_X A'[I] \{ U_X \} \rightarrow A'[I_X]$$

Likewise, $$\text{ReduceScatter}_X(A[I] \{U_X\}) \to A[I_X]$$ in the forward pass implies $$\text{AllGather}_{X}(A'[I_X]) \to A'[I]$$ in the backwards pass.

{% details For details on how AllGather and ReduceScatter are derivatives of eachother, click here. %}

This stems from the fact that broadcasts and reductions are transposes of eachother as linear operators, and AllGather and ReduceScatter are outer products (also known as [Kronecker products](https://en.wikipedia.org/wiki/Kronecker_product)) of broadcast and reduce, respectively. Concretely, if we have a vector $x \in \mathbb{R}^n$, any number of devices $p \in \mathbb{N}$, and we let $u = (1, \ldots, 1) \in \mathbb{R}^p$, we can define broadcast and reduce in the following way, which should match your intuitive understanding of them:

$$
\begin{align*}
\text{broadcast} &: \mathbb{R}^n \rightarrow \mathbb{R}^{p n} \\
\text{broadcast} &= u \otimes \mathbf{I}_n \\
\text{reduce} &: \mathbb{R}^{p n} \rightarrow \mathbb{R}^n \\
\text{reduce} &= u^T \otimes \mathbf{I}_n
\end{align*}
$$

Let's see how this looks in an example, where $n = 1$, $p = 2$. If $x = (7)$, we have $$\text{broadcast}(x) = \left(\begin{pmatrix} 1 \\ 1 \end{pmatrix} \otimes \begin{pmatrix} 1 \end{pmatrix}\right) x = \begin{pmatrix} 1 \\ 1 \end{pmatrix} x = \begin{pmatrix}  7\\  7  \end{pmatrix} \in \mathbb{R}^{p n}$$. This matches what we'd expect, broadcasting a vector in $\mathbb{R}^n$ to $\mathbb{R}^{pn}$. Now letting $y = (8, 9)$, we have $$\text{reduce}(y) = \left(\begin{pmatrix} 1 & 1 \end{pmatrix} \otimes \begin{pmatrix} 1\end{pmatrix}\right) y = \begin{pmatrix} 1 & 1  \end{pmatrix} \begin{pmatrix}  8 \\ 9  \end{pmatrix} = \begin{pmatrix}   17    \end{pmatrix}$$. This again matches what we'd expect, reducing a vector in $\mathbb{R}^{p n}$ to a vector in $\mathbb{R}^{n}$. Since $(A \otimes B)^T = A^T \otimes B^T$ for any two matrices $A$ and $B$, we see that $\text{reduce} = \text{broadcast}^T$. We recover AllGather and ReduceScatter as the following outer products:

$$
\begin{align*}
\text{AllGather} &: \mathbb{R}^{p n} \rightarrow \mathbb{R}^{p^2 n} \\
\text{AllGather} &= \text{broadcast} \otimes \mathbf{I}_p \\
\text{ReduceScatter} &= \mathbb{R}^{p^2 n} \rightarrow \mathbb{R}^{p n} \\
\text{ReduceScatter} &= \text{reduce} \otimes \mathbf{I}_p
\end{align*}
$$

Here we think of $\mathbb{R}^{p^2 n}$ as $\mathbb{R}^{p \times p n}$, so one $\mathbb{R}^{p n}$ vector for each of our $p$ devices. We suggest playing around with small examples, say $n = 2$, $p = 3$, to see what these operators look like as matrices. Using the same transposition property, we once more obtain $\text{AllGather}^T = \text{ReduceScatter}$, and of course $\text{ReduceScatter}^T = \text{AllGather}$. This transposition will arise during backpropagation, since if we have $y = Ax$ for some linear operator $A$, such as AllGather or ReduceScatter, then during backpropagation we will have the derivative of the loss with respect to $y$, $\frac{\partial L}{\partial y}$, and we obtain $\frac{\partial L}{\partial x}$ as $\frac{\partial L}{\partial x} = A^T \frac{\partial L}{\partial y}$. This shows how the derivative of AllGather will be ReduceScatter, and viceversa.

{% enddetails %}

Turning an AllReduce into an AllGather and ReduceScatter also has the convenient property that we can defer the final AllGather until some later moment. Very commonly we'd rather not pay the cost of reassembling the full matrix product replicated across the devices. Rather we'd like to preserve a sharded state even in this case of combining two multiplicands with sharded contracting dimensions:

$$A[I, J_X] \cdot B[J_X, K] \rightarrow C[I, K_X]$$

In this case, we can also perform a ReduceScatter instead of an AllReduce, and then optionally perform the AllGather at some later time, i.e.

$$\begin{align*}
A[I, J_X] \cdot_{LOCAL} B[J_X, K] \rightarrow &\ C[I, K] \{ U_X \} \\
\textbf{ReduceScatter}_{X,K} C[I, K] \{ U_X \} \rightarrow &\ C[I, K_X]
\end{align*}$$

Note that ReduceScatter *introduces* a sharded dimension, and so has a natural freedom to shard along either the **I** or **K** named dimensions in this case. We generally need to choose *which* named dimension to introduce a new sharding to when using a ReduceScatter (though the choice is usually forced by the larger modeling context). This is why we use the syntax **ReduceScatter<sub>X,K</sub>** to specify the axis to shard.

### How to overlap matmul communication with compute

As we discussed in [Part 1](../roofline), we generally assume we can always overlap communication with some useful computation if the comms are fast enough. The collectives in this section generally can be overlapped with the matrix multiplication compute itself, but doing so is non-trivial. The algorithm we use is something called a **collective matmul**, first described in [Wang et al.](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959). Here is a simplified animation of how this overlap can be implemented:

{% include figure.liquid path="assets/img/ag_matmul.gif" caption="<b>Figure:</b> an animation showing how a single sharded matrix-vector product can be overlapped with the resulting AllReduce (case 3 above). A full matmul is composed of multiple matrix-vector products." %}

To put it simply, we can do the matmul for one chunk of the matrix while starting the ring reduction for previous chunks. In some cases we can also tile over the batch dimension or matrix input dimension. We work through a simple JAX implementation in [Part 10](../jax-stuff) and [the Mosaic docs](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html) also gives a good example on GPU. We encourage you to implement a version of this at some point. 

## What Have We Learned?

* The sharding of an array is specified by a **Mesh** that names the physical, hardware axes of our TPU mesh and a **Sharding** that assigns mesh axis names to the logical axes of the array.
  * For example, **A**[I<sub>XY</sub>, J] describes an abstract array **A** with its first dimension sharded along two mesh axes X and Y. Combined with Mesh(mesh_shape=(4, 8), axis_names=('X', 'Y')) or the abbreviated Mesh({'X': 4, 'Y': 8}), this tells us our array is sharded 32 ways along the first dimension.

* **Arithmetic with sharded arrays works exactly like with unsharded arrays unless you perform a contraction along a sharded axis**. In that case, we have to introduce some communication. We consider four cases:

  1. *Neither array is sharded along the contracting dimension*: no communication is needed.
  2. *One array is sharded along the contracting dimension* (or the contracting dimensions are sharded along different axes): we AllGather one of the inputs before performing the operation.
  3. *Both arrays are identically sharded along the contracting dimension:* we multiply the shards locally then perform an AllReduce or ReduceScatter.
  4. *Both arrays are sharded along the same mesh axis along a non-contracting dimension:* we AllGather one of the inputs first.

* TPUs use roughly **4 core communication primitives**:
  1. AllGather: $[A_X, B] \to [A, B]$
  2. ReduceScatter: $[A, B] \\{U_X\\} \to [A, B_X]$
  3. AllToAll: $[A, B_X] \to [A_X, B]$
  4. AllReduce: $[A_X, B]\\{U_Y\\} \to [A_X, B]$ (technically not a primitive since it combines a ReduceScatter + AllGather)

{% include figure.liquid path="assets/img/all-collectives.png" class="img-fluid" %}

* The cost and latency of each of these operations **doesn't depend on the size of the axis (as long as they're bandwidth bound)**, but only on the size of the input arrays and the bandwidth of the link. For a unidirectional AllGather/ReduceScatter:

$$T_{\text{comm per AllGather or ReduceScatter}} = \frac{\text{Data volume}}{\text{bandwidth}} \cdot \frac{\text{Axis} - 1}{\text{Axis}}
\longrightarrow \frac{\text{Data volume}}{\text{bandwidth (bidirectional)}}$$

* An AllReduce is composed of a ReduceScatter followed by an AllGather, and thus has 2x the above cost. An AllToAll only has to pass shards part-way around the ring and is thus ¼ the cost of an AllGather. Here's a summary:

| Operation         | Description                                                                                                        | Syntax                           | Runtime                                          |
| :---------------- | :----------------------------------------------------------------------------------------------------------------- | :------------------------------- | :----------------------------------------------- |
| **AllGather**     | Gathers all the shards of a sharded array along an axis, removing a subscript.                                     | $[A_X, B] \to [A, B]$            | bytes / (bidirectional ICI bandwidth * num_axes) |
| **ReduceScatter** | Sums a partially summed array along an axis and shards it along another axis (adding a subscript).                 | $[A, B] \\{U_X\\} \to [A_X, B]$  | Same as AllGather                                |
| **AllReduce**     | Sums a partially summed array along an axis. Removes a { U<sub>x</sub> }. Combines an AllGather and ReduceScatter. | $[A_X, B]\\{U_Y\\} \to [A_X, B]$ | 2 * AllGather                                    |
| **AllToAll**      | Gathers (replicates) an axis and shards a different dimension along the same axis.                                 | $[A, B_X] \to [A_X, B]$          | AllGather / 4 for a bidirectional ring           |

## Some Problems to Work

*Here are some instructive problems based on content in this section. We won't include all answers at the moment but we'll write up more answers as we can.*

**Question 1 [replicated sharding]**: An array is sharded $A[I_X, J, K, \ldots]$ (i.e., only sharded across $X$), with a mesh `Mesh({'X': 4, 'Y': 8, 'Z': 2})`.  What is the ratio of the total number of bytes taken up by $A$ across all chips to the size of one copy of the array?

{% details 답을 보려면 여기를 클릭하세요. %}

Our array is only sharded along X, which has size 4, so effectively each shard has size $[I / 4, J, K, \ldots] = \text{sizeof}(A) / 4$. Since our array is replicated across Y and Z, the total size is $Y \cdot Z \cdot \text{sizeof}(A)$, so the ratio of total size to single chip size is $Y \cdot Z \cdot \text{sizeof}(A) / \text{sizeof}(A) = 16$.

{% enddetails %}

**Question 2 [AllGather latency]**: How long should $\text{AllGather}_X([B_X, D_Y])$ take on a TPUv4p 4x4x4 slice with mesh `Mesh({'X': 4, 'Y': 4, 'Z': 4})` if $B=1024$ and $D=4096$ in bfloat16? How about $$\text{AllGather}_{XY}([B_X, D_Y])$$? How about $$\text{AllReduce}_Z([B_X, D_Y] \{U_Z \})$$?

{% details 답을 보려면 여기를 클릭하세요. %}

We have a wraparound link on all axes because we have a full `4x4x4` cube, so we have 9e10 bidirectional bandwidth to work with.

1. Because we're just gathering over one axis and the other is sharded, we're effectively gathering $2BD / Y$ bytes over 1 axis. *If you think about just a single shard along the Y-axis, the AllGather along X looks like an unsharded AllGather with 1 / Y of the bytes.* Since our ICI bandwidth for TPU v4p is 9e10 bytes/second bidirectional, this will take $2BD / (\text{9e10} \cdot Y) = 2 \cdot 1024 \cdot 4096 / (\text{9e10} \cdot 4) = 23 \mu s$.

2. We have twice the bandwidth as before but we're AllGathering the full array, so `T = 2BD / (2 * W) = 2*1024*4096 / (2 * 9e10) = 46us`. This is far from the latency bound of 4us (1us per hop), so we're fine.

3. The cost of an AllReduce is twice that of an AllGather. Each shard has size $2BD / (X * Y)$, so the cost is about $4BD / (X * Y * W)$, or roughly `4 * 1024 * 4096 / (16 * 9e10) = 11.6us`.

{% enddetails %}

**Question 3 [latency-bound AllGather]**: Let's say we're performing an $\text{AllGather}_X([B_X])$ but $B$ is very small (say 128). How long should this take on a TPUv4p 4x4x4 slice with mesh `Mesh({'X': 4, 'Y': 4, 'Z': 4})` in bfloat16? *Hint: you're probably latency bound.*

{% details 답을 보려면 여기를 클릭하세요. %}

Our array in bfloat16 uses only 256 bytes total, and only 64 per device. Since we have an axis of size 4 on a TPU v4p, we have a wraparound link, so we can send the array in both directions. With `4.5e10` of unidirectional bandwidth, each hop would take roughly `64 / 4.5e10 ~ 0`, so we're definitely latency bound. Counting the number of hops, we can do the full gather in only 2 hops, so roughly 2us a good estimate.

{% enddetails %}

**Question 4 [matmul strategies]**: To perform $X[B, D] \cdot_D Y[D_X, F] \to Z[B, F]$, in this section we tell you to perform $\text{AllGather}_X(Y[D_X, F])$ and multiply the fully replicated matrices (Case 2, *Strategy 1*). Instead, you could multiply the local shards like $X[B, D_X] \cdot_D Y[D_X, F] \to Z[B, F] \\{U_X\\}$ (Case 4, *Strategy 2*), and then $\text{AllReduce}_X(Z[B, F] \\{ U_X\\})$. How many FLOPs and comms does each of these perform? Which is better and why?

{% details 답을 보려면 여기를 클릭하세요. %}

Let's start with our baseline (*Strategy 1*). As we've shown, the cost of the AllGather is $2DF / W_\text{ici}$. Once we have the fully replicated arrays, the total compute time is $2BDF / C$ (where $C$ is our accelerator FLOPs/s, since each TPU does the same FLOPs). So we have

$$T_\text{total (Strategy 1)} = \max\left(\frac{2BDF}{C}, \frac{2DF}{W_\text{ici}}\right)$$

By comparison, the new strategy (Strategy 2) does an AllReduce over $2BF$ bytes, which has cost $4BF / W_\text{ici}$ but does $1 / X$ fewer FLOPs (since the computation is sharded). This means we do $2\cdot B\cdot D\cdot F / X$ FLOPs and the resulting AllReduce communicates $$2 \cdot 2 \cdot B \cdot F$$ bytes in bfloat16. Thus, our total time for *Strategy 2* (no AllGather, just an AllReduce later on) is roughly

$$T_\text{total} = \max\left(\frac{2BDF}{X \cdot C}, \frac{4BF}{W_\text{ici}}\right)$$

The question is: *which of these is bigger?* Strategy (2) is compute bound when $D / (X \cdot C) > 2 / W_\text{ici}$, or when $D / 2X > C / W_\text{ici} \approx 2550 \rightarrow X < D / (2 * 2550)$. We might reasonably expect $D \approx 8k$, so this would mean roughly $X < 2$ which is unlikely – hence we're basically always comms bound with Strategy 2. With the baseline (Strategy 1), we're comms bound when $$B < C / W_\text{ici} = 2550$$ which is often but not always true.

So if $B < 2550$, we're comms-bound in both cases and we have

$$T_\text{comms for Strategy 2} < T_\text{comms for Strategy 1} \Leftrightarrow \frac{4BF}{W_\text{ici}} < \frac{2DF}{W_\text{ici}}$$

which is true when $D > 2B$ where $2B < 5100$. This is often true, so Strategy 2 can sometimes be better if our batch is small. When our batch is large ($B > 2550$), we have

$$T_\text{comms for Strategy 2} < T_\text{math for Strategy 1} \Leftrightarrow \frac{4BF}{W_\text{ici}} < \frac{2BDF}{C}$$

This is true when $2 / W_\text{ici} < D / C$, or when $D > 2 * 2550 = 5100$, which is usually true for large models. So this alternative strategy is typically better for large models, unless $D$ is small.

*Why don't we always do this?* Well, in practice we may do this sometimes, but it's typically rare to have the contracting dimension of one of the inputs to a matmul sharded along a axis that the other input isn't sharded over. For instance, if we're doing FSDP (explained in [Section 5](../training)), we'll shard our parameters over the data dimension but our activations will _also be sharded along data_. So in this sense this doesn't show up much.

{% enddetails %}

**Question 5 [minimum latency]**: Let's say I want to do a matmul $A[I, J] \cdot_J B[J, K] \to C[I, K]$ on a TPUv5p 4x4x4 with the lowest possible latency. Assume the inputs can be sharded arbitrarily but the result should be fully replicated. How should my inputs be sharded? What is the total FLOPs and comms time?

{% details Click here for the (partial) answer. %}

We won't provide a full answer here, but we'll start by describing the four most likely options:

1. $A[I_{XYZ}, J] \cdot B[J, K]$ + AG at the end
2. $A[I, J] \cdot B[J, K_{XYZ}]$ + AG at the end
3. $A[I, J_{XYZ}] \cdot B[J_{XYZ}, K]$ + AR at the end
4. $A[I, J] \cdot B[J, K]$ (fully replicated)

We could also consider sharding different axes along different mesh axes, but that isn't likely to change the final cost. For all but (4), the total FLOPs per TPU is the same, but comms are different for each. We then simply need to calculate the comms cost for each and see which is lowest. The TLDR is that (1) and (2) are equally good.

{% enddetails %}

**Question 6:** Let's say we want to perform $A[I_X, J_Y] \cdot_J B[J_Y, K] \to C[I_X, K]$ on TPUv5e 4x4. What communication do we perform? How much time is spent on communication vs. computation?

* What about $A[I_X, J] \cdot_J B[J_X, K_Y] \to C[I_X, K_Y]$? This is the most standard setting for training where we combine data, tensor, and zero sharding. 
* What about $A[I_X, J] \cdot_J B[J, K_Y] \to C[I_X, K_Y]$? This is standard for inference, where we do pure tensor parallelism (+data).

**Question 7:** A typical Transformer block has two matrices $W_\text{in}[D, F]$ and $W_\text{out}[F, D]$ where $F \gg D$. Say we have a batch size B. Then the full block is $In[B, D] \cdot W_\text{in}[D, F]. \cdot W_\text{out}[F, D]$. Let's pick $D=8192$, $F=32768$, and $B=128$ and assume everything is in bfloat16. Assume we're running on a TPUv5e 2x2 slice but let's pretend each TPU only has 300MB of free memory. How should In, $W_\text{in}$, $W_\text{out}$, and Out be sharded to stay below the memory limit while minimizing overall time? How much time is spent on comms and FLOPs? *Hint: the final output doesn't need to be fully replicated, but it should be sharded the same as the input so the "layer" can be repeated.*

{% details Click here for the (partial) answer. %}

First let's think about memory. Each of our two big matrices uses `2 * 8192 * 32768 = 536MB`. Our activations `In` have size `128 * 8192 = 1MB` (small enough not to worry about). Since we only have 300MB of spare memory in each device, we clearly need to shard our matmuls.

1. $In[B_X, D] * W_\text{in}[D_{XY}, F] * W_\text{out}[F, D_{XY}] \rightarrow Out[B, D]$ (this is often called FSDP)
2. $In[B, D_{XY}] * W_\text{in}[D, F_{XY}] * W_\text{out}[F_{XY}, D] \rightarrow Out[B, D_{XY}]$ (this is called tensor parallelism)

The first is pretty bad because we need to AllGather our big weights or our activations first. The second requires an AllGather at the beginning and a ReduceScatter at the end (which is cheaper than an AllReduce). I'll leave it as an exercise to do the rest of the math.

{% enddetails %}

**Question 8 [challenge]**: Using the short code snippet above as a template, allocate a sharded array and benchmark each of the 4 main communication primitives (AllGather, AllReduce, ReduceScatter, and AllToAll) using pmap or shard_map. You will want to use `jax.lax.all_gather`, `jax.lax.psum`, `jax.lax.psum_scatter`, and `jax.lax.all_to_all`. Do you understand the semantics of these functions? How long do they take?

**Question 9 [another strategy for sharded matmuls?]**: [Above](#case-2-one-multiplicand-has-a-sharded-contracting-dimension) we claimed that when only one input to a matmul is sharded along its contracting dimension, we should AllGather the sharded matrix and perform the resulting contracting locally. Another strategy you might think of is to perform the sharded matmul and then AllReduce the result (as if both inputs were sharded along the contracting dimension), i.e. $A[I, J_X] *_J B[J, K] \to C[I, K]$ by way of

1. $C[I, K] \\{ U_X \\} = A[I, J_X] \cdot B[J_X, K]$
2. $C[I, K] = \text{AllReduce}(C[I, K] \\{ U_X\\})$

Answer the following:

1. Explicitly write out this algorithm for matrices $A[N, M]$ and $B[M, K]$, using indices to show exactly what computation is done on what device.  Assume $A$ is sharded as $A[I, J_X]$ across ND devices, and you want your output to be replicated across all devices.
2. Now suppose you are ok with the final result not being replicated on each device, but instead sharded (across either the N or K dimension).  How would the algorithm above change?
3. Looking purely at the communication cost of the strategy above (in part (b), not (a)), how does this communication cost compare to the communication cost of the algorithm in which we first AllGather A and then do the matmul?

{% details 답을 보려면 여기를 클릭하세요. %}


1. First compute the outer products, storing the result in $$O[N, K]: o_{kj} = \sum_i a_{ki} b_{ij}$$. Note that the repeated index is not the one being contracted, as we are doing an outer product. Here the sum ranges across the set of i values stored on the particular device we are using. So, for example, if we have a contracting axis of size 16, and 4 devices, then on device 0, i would range from {0, 1, 2, 3}; on device 1, i would range from {4, 5, 6, 7}; on device 2, i would range from {8, 9, 10, 11}; and on device 3, i would range from {12, 13, 14, 15}. Then AllReduce the partial-sums of $O[N, K]$ which live on each device, to form the full $O[N, K]$.
2. Instead of doing an AllReduce in step 2, we could get away with a cheaper ReduceScatter, along either axis: $[N, K] \\{ U_X \\} \to [N_X, K]$ or $[N, K] \\{ U_X \\} \to [N, K_X]$.
3. As described in the main text above, the cost of doing an AllGather (when we are throughput-bound) is the same as that of a ReduceScatter; it is simply given by the size of the full matrix we are processing.  So in the gather-then-matmul algorithm, this scales as $NM$ (since we are $\text{AllGather}$-ing $A$); in the matmul-then-reduce-scatter algorithm, this scales as NK (since we are reduce-scattering $O$). So the communication cost ratio of the two algorithms is `M/K`.

{% enddetails %}

**Question 10: Fun with AllToAll:** In the table above, it was noted that the time to perform an AllToAll is a factor of 4 lower than the time to perform an AllGather or ReduceScatter (in the regime where we are throughput-bound). In this problem we will see where that factor of 4 comes from, and also see how this factor would change if we only had single-direction ICI links, rather than bidirectional ICI links.

1. Let's start with the single-direction case first.  Imagine we have *D* devices in a ring topology, and  If we are doing either an AllGather or a ReduceScatter, on an N x N matrix *A* which is sharded as $A[I_X, J]$ (say $D$ divides $N$ for simplicity).  Describe the comms involved in these two collectives, and calculate the total number of scalars (floats or ints) which are transferred across **a single** ICI link during the entirety of this algorithm.
2. Now let's think about an AllToAll, still in the single-directional ICI case.  How is the algorithm different in this case than the all-gather case?  Calculate the number of scalars that are transferred across a single ICI link in this algorithm.
3. You should have found that the ratio between your answers to part (a) and part (b) is a nice number.  Explain where this factor comes from in simple terms.
4. Now let's add bidirectional communication. How does this affect the total time needed in the all-gather case?
5. How does adding bidirectional communication affect the total time needed in the AllToAll case?
6. Now simply explain the ratio between AllGather time and AllToAll time in a bidirectional ring.

{% details 답을 보려면 여기를 클릭하세요. %}

(1) **Solution:** The process is simple: in each step of the algorithm, each device will send a single-shard "strip” of the matrix (totalling $$\frac{N}{D} \times N$$ elements in size) to its nearest neighbor. This occurs $$D-1$$ times, since each shard needs to be communicated to all of the devices except the one it starts out on. So in total, $$\frac{N^2(D-1)}{D}$$ scalars are transferred by each device, i.e. flow across a single ICI link.

**Answer:** $$N^2 (1-\frac{1}{D})$$, or simply $$N^2$$ when $$D >> 1$$.

(2) **Solution:** The key difference between an AllToAll and an AllGather, from the perspective of communications, is that in an AllToAll, the entirety of the shard that lives on a particular device does not need to be communicated to every other device. Imagine the shard stored on a particular device (call it device 0) is $$[A, B, C, D]$$ (here A,B,C,D are matrices and we are imagining a ring with 4 devices for illustration). Now the matrix $$A$$ does not need to be communicated anywhere, the matrix $$B$$ needs to end up on device 1; matrix $$C$$ ends up on device 2; and matrix $$D$$ ends up on device 3. So in the first step of the algorithm, we send $$B$$, $$C$$, and $$D$$ to device 1; in the next step, device 1 sends $$C$$ and $$D$$ onwards to device 2; in the final step, device 2 sends just $$D$$ on to device 3. The total number of parameters transferred in this case is $$(\text{size of A/B/C/D}) * (3 + 2 + 1)$$. The size of A/B/C/D is (in the general case now) $$\frac{N^2}{D^2}$$, and again in the general case the $$(3 + 2 + 1)$$ term becomes $$((D-1) + (D-2) + … + 1)$$, or $$\frac{(D)(D-1)}{2}$$. So the total number of bytes transferred across a single ICI link is $$\frac{N^2(D-1)}{D \times 2}$$.

**Answer:** $$\frac{N^2}{2}(1-\frac{1}{D})$$, or simply $$\frac{N^2}{2}$$ when $$D >> 1$$.

(3) **Solution:** The factor is simply $$\frac{1}{2}$$, i.e. an AllToAll is half as costly as an all-gather/ReduceScatter on a unidirectional ring topology. Looking over the derivations above, this ultimately came from the fact that in the all-gather case, we are transferring the same sized block each of $$(D-1)$$ times, i.e. we're doing the sum $$ \text{tiny block size} * (D + D + D + … + D)$$, whereas in the AllToAll case, we're doing the sum $$\text{tiny block size} * (D + D-1 + D-2 + … + 1)$$. The factor of two thus essentially comes from the fact that $$1 + 2 + \ldots + n = n(n+1)/2$$.

(4) **Solution**:  The total number of scalars that any one link has to carry now reduces by a factor of 2, since in a bidirectional ring, each "sharded strip” can be sent two ways simultaneously.

(5) **Solution**: In this case, we win a factor of 4 compared to the unidirectional case.  This is easiest to see by considering the fate of each of the size-(N2/D2) blocks in a single sharded strip, say the one which originates on device 0.  Instead of (as in the unidirectional case) sending one of these blocks a distance of D-1, another block a distance D - 2, etc. all the way to 1, we now divide the strip into blocks which move right or left, moving a maximum distance of floor(D/2).  So the corresponding sum now becomes $$D/2 + D/2 - 1 + D/2 - 2 + … = D/2 \cdot (D/2+1)/2$$, or $$D^2/8$$ in the limit of large $$D$$.  Compare this to $$D^2/2$$ in the unidirectional case, and we see that we've won a factor of 4.

(6) **Solution:** In a unidirectional ring, we saw that the AllToAll time was already twice as fast as the all-gather time; this comes from the fact that we don't need to send our full strip to every single device.  Then, when we added bidirectionality, we saw that it was a 4x win for AllToAll, and only a 2x win for all-gathers.  Putting these ratios together, we get our sought after factor of 4.

{% enddetails %}

<h3 markdown=1 class="next-section">That's it for Part 3! For Part 4 (about Transformer math), click [here](../transformers)!</h3>
