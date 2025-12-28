---
layout: distill
title: "Programming TPUs in JAX"
# permalink: /main/
description: "JAX를 사용하여 TPU를 효율적으로 프로그래밍하는 방법! 이 섹션의 많은 부분은 <a href='https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html'>여기</a>에서 가져왔습니다. <a href='https://colab.sandbox.google.com/'>Google Colab</a>에서 무료 TPU로 이 섹션의 코드 예제를 실행할 수 있습니다."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 10

previous_section_url: "../profiling"
previous_section_name: "Part 9: Profiling"

next_section_url: "../conclusion"
next_section_name: "Part 11: Conclusions"

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
  - name: Yash Katariya
    url: https://x.com/yashk2810
  - name: Yash Katariya
    url: https://x.com/yashk2810
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "How Does Parallelism Work in JAX?"
  - subsections:
    - name: "Auto sharding mode"
    - name: “Explicit sharding mode”
    - name: "Manual sharding mode via shard_map"
    - name: "Auto sharding mode"
    - name: “Explicit sharding mode”
    - name: "Manual sharding mode via shard_map"
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

## How Does Parallelism Work in JAX?

JAX는 다중 디바이스 프로그래밍에 대해 세 가지 학파를 지원합니다:

1. **컴파일러가 운전대를 잡아라!** XLA 컴파일러가 자동으로 배열을 분할하고 주어진 프로그램을 용이하게 하기 위해 어떤 통신을 추가할지 결정하게 합니다. 이를 통해 단일 디바이스에서 실행되는 프로그램을 아무런 변경 없이 수천 개의 칩에서 자동으로 실행할 수 있습니다.
2. **JAX가 운전대를 잡아라!** 자동 병렬화는 훌륭하지만 때때로 컴파일러가 엉뚱한 짓을 합니다. Explicit sharding은 평소처럼 단일 디바이스 코드를 작성하되, (컴파일러가 아닌) JAX가 샤딩 전파를 처리하게 합니다. 이는 사용자가 원하는 바가 불분명할 때 JAX가 설명을 요청할 수 있음을 의미합니다.
3. **그냥 내가 의도한 대로 쓰게 해줘, 제기랄!** 컴파일러는 좋지만 때로는 잘못된 일을 하여 의도하지 않은 통신을 추가합니다. 때로는 실행하려는 통신을 정확히 명시하고 싶을 때가 있습니다.

| Mode | View? | Explicit sharding? | Explicit Collectives? |
|:---:|:---:|:---:|:---:|
| Auto | Global | ❌ | ❌ |
| Explicit | Global | ✅ | ❌ |
| Manual | Per-device | ✅ | ✅ |

이에 맞춰 JAX는 각 모드에 대한 API를 제공합니다:

1. `jax.jit` (`Auto` 메시 축 사용)은 기존 JAX 함수를 가져와 샤딩된 입력으로 호출할 수 있게 해줍니다. 그러면 JAX는 프로그램을 자동으로 병렬화하는 XLA의 [Shardy](https://openxla.org/shardy) 컴파일러를 사용합니다. XLA는 기존 연산을 용이하게 하기 위해 필요할 때 통신(AllGather, ReduceScatter, AllReduce 등)을 자동으로 추가합니다. 완벽하지는 않지만, 일반적으로 코드 변경 없이 프로그램을 임의의 수의 칩으로 자동 확장하는 데 꽤 괜찮은 작업을 수행합니다.
2. `Explicit` 메시 축을 사용하는 `jax.jit`은 (1)과 비슷해 보이지만 XLA 대신 JAX가 샤딩 전파를 처리하게 합니다. 즉, 배열의 샤딩은 실제로 JAX 타입 시스템의 일부이며, JAX는 모호한 통신을 감지하면 오류를 발생시키고 사용자가 이를 해결하도록 합니다.
3. `jax.shard_map`은 더 수동적인 대응물입니다. 프로그램의 디바이스 로컬 뷰를 얻게 되며 원하는 모든 통신을 명시적으로 작성해야 합니다. 샤딩된 배열이 있고 각 디바이스에 전체를 원하나요? `jax.lax.all_gather`를 추가하세요. 디바이스 전체에 걸쳐 배열을 합산하고 싶나요? `jax.lax.psum`(AllReduce)을 추가하세요. 프로그래밍은 더 어렵지만 원하지 않는 일이 발생할 가능성은 훨씬 적습니다.

<h3 id="auto-sharding-mode">Auto sharding mode</h3>
<h3 id="auto-sharding-mode">Auto sharding mode</h3>

jax.jit은 JAX 내부에서 두 가지 역할을 합니다. 이름에서 알 수 있듯이 Python 함수를 바이트코드(XLA/HLO/LLO를 통해)로 "just-in-time" 컴파일하여 더 빠르게 실행되도록 합니다. 그러나 입력이 샤딩되었거나 사용자가 `in_sharding` 또는 `out_sharding`을 지정한 경우, XLA가 여러 디바이스에 계산을 분산하고 필요에 따라 통신을 추가할 수도 있게 해줍니다. 예를 들어, 다음은 jax.jit을 사용하여 샤딩된 matmul을 작성하는 방법입니다:

```python
import jax
import jax.numpy as jnp

# TPU v5e 4x2에서 실행 중. 하드웨어의 두 물리적 축에 이름을 할당합니다.
mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=('X', 'Y'))

# 이것은 JAX에게 모든 연산에 이 메시를 사용하도록 지시하므로 PartitionSpec P만 지정하면 됩니다.
jax.set_mesh(mesh)

# 디바이스에 걸쳐 샤딩된 행렬 W와 입력 활성화 In을 생성합니다.
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('X', 'Y')))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('Y', None)))

def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

# 여기서 샤딩된 matmul 함수를 명시적으로 컴파일할 수 있습니다. 
# 이것은 필요한 모든 통신(예: matmul 후의 AllReduce)을 추가합니다.
jit_matmul = jax.jit(matmul_square, out_shardings=jax.P('X', None)).lower(In, W).compile()

out = jit_matmul(In, W)
```

이것은 어떤 샤딩으로든 자동으로 실행되며 계산을 디바이스에 분할합니다. **하지만 하드웨어 수준에서는 실제로 무슨 일이 일어나고 있을까요?**

1. 먼저 디바이스에 걸쳐 샤딩된 In과 W를 생성합니다<d-footnote>우리가 이것을 어떻게 했는지 주목하세요. 이것은 특정 샤딩으로 배열을 생성하는 한 가지 방법입니다(즉, 생성 함수에 device 인수를 추가하여). 또 다른 방법은 `jnp.array(....)`로 정상적으로 배열을 생성한 다음 `jax.device_put(..., jax.P('x', 'y'))` 등을 수행하는 것입니다. 또 다른 방법은 원하는 배열을 생성하는 함수를 작성하고 원하는 `out_shardings`로 jit 컴파일하는 것입니다.</d-footnote>. W는 축약 차원을 따라 2방향으로 샤딩되고, In은 4방향으로(축약 및 출력 차원 모두를 따라) 샤딩됩니다. 이는 W[D<sub>Y</sub>, F] 및 In[B<sub>X</sub>, D<sub>Y</sub>] 샤딩에 해당하며, 일종의 모델 및 데이터 병렬 처리입니다.
2. 이것을 로컬에서(즉, 한 디바이스에서) 실행한다면 `matmul_square`는 단순히 입력을 제곱하고 간단한 matmul을 수행할 것입니다. 그러나 `out_shardings`를 `P('X', None)`으로 지정했기 때문에 출력은 배치를 따라 샤딩되지만 모델 차원에 걸쳐 복제되며 계산하려면 AllReduce가 필요합니다.

이전 섹션의 표기법을 사용하면 이는 아마도 다음과 같은 작업을 수행할 것입니다.

1. Out[B<sub>X</sub>, F] { U<sub>Y</sub> } = In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F]
1. Out[B<sub>X</sub>, F] { U<sub>Y</sub> } = In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F]
2. Out[B<sub>X</sub>, F] = **AllReduce**(Out[B<sub>X</sub>, F] { U<sub>Y</sub> })

`jax.jit`은 우리를 위해 이것을 자동으로 추가합니다! `jit_matmul.as_text()`로 HLO를 실제로 출력하여 다음 HLO를 볼 수 있습니다(대폭 축약됨):

```python
# 이 fusion은 샤딩된 입력과 행렬의 실제 matmul입니다
%fusion = bf16[2,8192]{1,0:T(4,128)(2,1)S(1)} fusion(bf16[2,1024]{1,0:T(4,128)(2,1)} %param, bf16[8192,1024]{1,0:T(8,128)(2,1)S(1)} %copy-done)

# 우리는 디바이스 전반에 걸쳐 부분 합산된 결과를 reduce합니다
ROOT %AllReduce = bf16[2,8192]{1,0:T(4,128)(2,1)} AllReduce(bf16[2,8192]{1,0:T(4,128)(2,1)S(1)} %fusion)
```

위에서 matmul (fusion)과 AllReduce를 볼 수 있습니다. 모양에 특히 주의하세요. `bf16[2, 1024]`는 활성화의 로컬 뷰입니다. `batch_size=8`이 4개의 디바이스에 걸쳐 분할되고 `d_model=2048`도 2방향으로 분할되기 때문입니다.

**이것은 꽤 마법 같습니다!** 프로그램이 아무리 복잡하더라도 [Shardy](https://openxla.org/shardy)와 jit은 모든 중간 활성화에 대한 샤딩을 찾고 필요에 따라 통신을 추가하려고 시도합니다. 그렇긴 하지만 Shardy에도 결함이 있습니다. 실수를 할 수 있습니다. 때때로 프로파일을 보면 뭔가 잘못되었다는 것을 알게 될 것입니다. 거대한 AllGather가 필요하지 않은데도 프로파일의 80%를 차지합니다. 이런 일이 발생하면 `jax.lax.with_sharding_constraint`로 중간 텐서에 명시적으로 주석을 달아 컴파일러를 수정하려고 시도할 수 있습니다. 예를 들어 두 개의 matmul을 사용하여 중간 활성화가 `y` 차원을 따라 샤딩되도록 강제할 수 있습니다(이것이 좋은 생각이라는 것은 아님):

```python
import jax
import jax.numpy as jnp

mesh = jax.make_mesh((4, 2), ('X', 'Y'))

def matmul(x, Win, Wout):
  hidden = jnp.einsum('bd,df->bf', x, Win)
  hidden = jax.lax.with_sharding_constraint(hidden, jax.P('x', 'y'))
  hidden = jax.lax.with_sharding_constraint(hidden, jax.P('x', 'y'))
  return jnp.einsum('bf,df->bd', hidden, Wout)
```

이것은 `jax.lax.with_sharding_constraint`를 통해 중간 샤딩을 제어하는 자동 분할 세계에서 JAX 병렬 프로그래밍의 약 60%를 차지합니다. 하지만 "컴파일러 간지럽히기(compiler tickling)"는 유명하게도 재미있는 프로그래밍 모델이 아닙니다. 모든 중간 변수에 주석을 달아도 여전히 올바른 결과를 얻을 수 있을지 알 수 없습니다. 대신 JAX 자체가 샤딩 전파를 처리하고 제어할 수 있다면 어떨까요?

<h3 id="explicit-sharding-mode">Explicit sharding mode</h3>

Explicit sharding (또는 “sharding in types”)은 자동 샤딩과 매우 유사해 보이지만 샤딩 전파가 JAX 수준에서 발생합니다! 각 JAX 연산에는 op 인수의 샤딩을 가져와 op 결과에 대한 샤딩을 생성하는 샤딩 규칙이 있습니다. `jax.typeof`를 사용하여 결과 샤딩을 볼 수 있습니다:

```python
import jax
import jax.numpy as jnp
import jax.sharding as shd

# TPU v5e 2x2에서 실행 중. 하드웨어의 두 물리적 축에 이름을 할당합니다.
mesh = jax.make_mesh(axis_shapes=(2, 2), axis_names=('X', 'Y'),
                     axis_types=(shd.AxisType.Explicit, shd.AxisType.Explicit))

# 이것은 JAX에게 모든 연산에 이 메시를 사용하도록 지시하므로 PartitionSpec P만 지정하면 됩니다.
jax.set_mesh(mesh)

x = jax.device_put(np.arange(16).reshape(8, 2), jax.P('X', 'Y'))

@jax.jit
def f(x):
  print(jax.typeof(x))  # bfloat16[8@X,2@Y]
  out = x * 2
  print(jax.typeof(out))  # bfloat16[8@X,2@Y]
  return out

f(x)
```

보시다시피 JAX는 입력(`x`)에서 출력(`x`)으로 샤딩을 전파했으며, 이는 `jax.typeof`를 통해 trace-time에 검사할 수 있습니다. 대부분의 연산의 경우 합리적인 선택이 하나뿐이기 때문에(예: elementwise op은 동일한 샤딩을 유지함) 이러한 규칙은 간단하고 분명합니다. 그러나 일부 연산의 경우 결과를 샤딩하는 방법이 모호하여 JAX가 trace-time 오류를 발생시키고 프로그래머에게 `out_sharding` 인수를 명시적으로 제공하도록 요청합니다(예: jnp.einsum, jnp.reshape 등). 충돌이 있는 또 다른 예를 보겠습니다:

```python
# 디바이스에 걸쳐 샤딩된 행렬 W와 입력 활성화 In을 생성합니다.
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, out_sharding=jax.P('X', 'Y'))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, out_sharding=jax.P('Y', None))

@jax.jit
def matmul_square(In, W):
  print(jax.typeof(In))  # bfloat16[8@X, 2048@Y]
  print(jax.typeof(W))  # bfloat16[2048@Y, 8192]
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

matmul_square(In, W)  # 이것은 오류를 발생시킵니다
```

이 코드는 `Contracting dimensions are sharded and it is ambiguous how the output should be sharded. Please specify the output sharding via the out_sharding parameter. Got lhs_contracting_spec=('Y',) and rhs_contracting_spec=('Y',)` 오류를 발생시킵니다.

이는 einsum의 출력이 어떻게 샤딩되어야 하는지 모호하기 때문에 아주 좋습니다. 출력 샤딩은 다음과 같을 수 있습니다:
* reduce-scatter를 유도할 P('X', 'Y') 또는
* all-reduce를 유도할 P('X', None)

Auto 모드와 달리 explicit 모드는 모호한 통신을 감지하면 오류를 발생시키고 사용자가 해결하도록 요구합니다. 따라서 여기서는 다음과 같이 할 수 있습니다:

```python
@jax.jit
def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W, out_sharding=jax.P('X', 'Y'))

out = matmul_square(In, W)
print(jax.typeof(out))  # bfloat16[8@X,8192@Y]
```

Auto 모드와 Explicit 모드는 `jax.sharding.auto_axes` 및 `jax.sharding.explicit_axes` API를 통해 구성할 수 있습니다. 자세한 내용은 [이 문서](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html)를 읽어보세요.

<h3 id="manual-sharding-mode-via-shard_map">shard_map: explicit parallelism control over a program</h3>
<h3 id="manual-sharding-mode-via-shard_map">shard_map: explicit parallelism control over a program</h3>

Shardy가 "컴파일러가 운전대를 잡아라" 모드라면, jax [shard_map](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html)은 모든 것을 여러분의 손에 맡깁니다. jax.jit에서처럼 입력의 샤딩을 지정하지만, 모든 통신을 명시적으로 작성합니다. `jax.jit`이 프로그램에 대한 글로벌 교차 디바이스 뷰를 남기는 반면, `shard_map`은 로컬 디바이스별 뷰를 제공합니다.

예를 들어 보겠습니다. 이 함수가 무엇을 하는지 추론해 보세요:<d-footnote>메시를 에뮬레이트하여 colab에서 직접 실행해보고 싶다면 `import jax; jax.config.update('jax_num_cpu_devices', 8)` 셀을 사용하면 됩니다.</d-footnote>

```python
import jax
import jax.numpy as jnp
import jax.sharding as shd

mesh = jax.make_mesh((2, 4), ('x', 'y'), (shd.AxisType.Explicit, shd.AxisType.Explicit))
jax.set_mesh(mesh)
mesh = jax.make_mesh((2, 4), ('x', 'y'), (shd.AxisType.Explicit, shd.AxisType.Explicit))
jax.set_mesh(mesh)

x = jnp.arange(0, 512, dtype=jnp.int32, out_sharding=jax.P(('x', 'y')))

# 이 함수는 배열의 1/8에서 작동합니다.
@jax.shard_map(in_specs=jax.P(('x', 'y')), out_specs=jax.P())
def slice_and_average(x):
  assert x.shape == (512 // 8,)
  return jax.lax.pmean(x[:4], axis_name=('x', 'y'))

out = slice_and_average(x)
out = slice_and_average(x)
assert out.shape == (4,)
```

**이것은 무엇을 합니까?** `slice_and_average`는 배열의 1/8을 가진 각 TPU에서 실행되며, 여기서 처음 4개 요소를 슬라이스하여 전체 메시에 걸쳐 평균을 냅니다. 즉, 사실상 `mean(x[:4], x[64:68], x[128:132], …)`을 수행하고 있습니다. JAX에서 달리 표현하기 쉽지 않은 연산이기 때문에 꽤 멋집니다.

**왜 jax.jit 대신 이것을 하나요?** `jax.jit`을 사용했다면 `slice_and_average`는 배열의 글로벌 뷰(전체 `[512,]` 배열)를 보았을 것입니다. 우리는 이 불균일한 슬라이스를 잘라낸 다음 XLA가 올바르게 해석해야 할 평균을 수행해야 했을 것입니다. XLA가 잘못된 통신을 추가하거나 혼란스러워했을 수 있습니다. 여기서는 로컬 뷰를 보고 필요한 통신만 작성합니다.

**Example [Collective Matmul]:** 좀 더 현실적인 예를 들어, 활성화가 초기에 모델 샤딩된 모델 병렬 처리를 구현한다고 가정해 봅시다. 즉, A[B<sub>X</sub>, D<sub>Y</sub>] \* W[D, F<sub>Y</sub>] -> Out[B<sub>X</sub>, F<sub>Y</sub>]. 순진하게는 A를 먼저 AllGather한 다음 로컬 행렬 곱셈을 수행하여 이를 수행할 것입니다:

1. A[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(A[B<sub>X</sub>, D<sub>Y</sub>])
1. A[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(A[B<sub>X</sub>, D<sub>Y</sub>])
2. Out[B<sub>X</sub>, F<sub>Y</sub>] = A[B<sub>X</sub>, D] *<sub>D</sub> W[D, F<sub>Y</sub>]

안타깝게도 이것은 통신과 계산을 중첩시킬 수 없기 때문에 나쁩니다. 이들을 중첩하는 것은 [Wang et al. 2023](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959)에 설명된 "collective matmul"로 수행할 수 있습니다. 알고리즘은 기본적으로 다음과 같습니다:

* 각 Y 샤드에 대해, A의 로컬 청크와 W의 로컬 청크의 matmul을 수행하여 `[B / X, F / Y]` 모양의 결과를 생성합니다. 동시에 다음 청크를 로컬로 얻도록 A를 순열(permute)하고, matmul을 수행하고, 결과를 합산합니다.

`jax.shard_map`으로 아주 쉽게 구현할 수 있습니다:

```python
import functools

import jax
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np

# 이것은 TPU v5e-8 런타임에서 실행되도록 의도되었습니다. 이것을 얻을 수 없다면
# jax.config.update('jax_num_cpu_devices', 8)을 설정해 보세요.
#
mesh = jax.make_mesh(axis_shapes=(2, 4), axis_names=('X', 'Y'),
                     axis_types=(shd.AxisType.Explicit, shd.AxisType.Explicit))
jax.set_mesh(mesh)

B, D, F = 1024, 2048, 8192
A = jnp.arange(np.prod((B, D))).reshape((B, D))
W = jnp.arange(np.prod((D, F))).reshape((D, F))

A = jax.device_put(A, jax.P('X', 'Y'))
W = jax.device_put(W, jax.P(None, 'Y'))
A = jax.device_put(A, jax.P('X', 'Y'))
W = jax.device_put(W, jax.P(None, 'Y'))

@functools.partial(jax.jit, out_shardings=jax.P('X', 'Y'))
@functools.partial(jax.jit, out_shardings=jax.P('X', 'Y'))
def matmul(lhs, rhs):
  return lhs @ rhs

def collective_matmul_allgather_lhs_contracting(lhs, rhs):
  # lhs는 루프 피연산자이고; rhs는 로컬 피연산자입니다
  axis_size = jax.lax.axis_size('Y')  # 이 예제의 경우 axis_size = 4
  idx = jax.lax.axis_index('Y')

  chunk_size = lhs.shape[1]
  assert rhs.shape[0] % chunk_size == 0

  def f(i, carrys):
    accum, lhs = carrys
    rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
    # 청크에 대한 Matmul
    update = lhs @ rhs_chunk
    # 왼쪽으로 순환 시프트
    lhs = jax.lax.ppermute(
        lhs,
        axis_name='Y',
        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)]
    )
    return accum + update, lhs

  accum = jnp.zeros((lhs.shape[0], rhs.shape[1]), dtype=lhs.dtype)
  accum = jax.lax.pvary(accum, ('X', 'Y'))
  accum = jax.lax.pvary(accum, ('X', 'Y'))
  accum, lhs = jax.lax.fori_loop(0, axis_size - 1, f, (accum, lhs), unroll=True)

  # lhs를 발견한 상태로 두기 위해 최종 순열 후 마지막 청크 계산
  i = axis_size - 1
  rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
  update = lhs @ rhs_chunk
  return accum + update

jit_sharded_f = jax.jit(jax.shard_map(
  collective_matmul_allgather_lhs_contracting,
  in_specs=(jax.P('X', 'Y'), jax.P(None, 'Y')), out_specs=jax.P('X', 'Y')))
jit_sharded_f = jax.jit(jax.shard_map(
  collective_matmul_allgather_lhs_contracting,
  in_specs=(jax.P('X', 'Y'), jax.P(None, 'Y')), out_specs=jax.P('X', 'Y')))

shmapped_out = jit_sharded_f(A, W)
expected_out = matmul(A, W)

np.testing.assert_array_equal(shmapped_out, expected_out)
```

이것은 꽤 깔끔합니다! 이것을 벤치마크해 보면 또한 훨씬 빠르다는 것을 알 수 있습니다! [여기](https://imgur.com/a/e9I6SrM) 시작 부분에 큰 차단 AllGather가 있어 311us가 걸리는 기본 jit matmul의 프로파일이 있습니다:

{% include figure.liquid path="assets/img/not-overlapped.png" class="img-fluid" %}

그리고 [여기](https://imgur.com/a/21iy0Sv) 244us가 걸리는 위의 버전이 있습니다. 프로파일에 AllGather가 없는 것을 볼 수 있습니다. 모두 유용한 작업입니다! FLOPs 활용률도 훨씬 높습니다.

{% include figure.liquid path="assets/img/overlapped.png" class="img-fluid" %}

또한 축약 차원에 샤딩이 없는 matmul 시간이 [224us](https://imgur.com/a/i3gNKfq)이므로 여기서 샤딩되지 않은 기준선에 놀랍게 가깝다는 점에 주목할 가치가 있습니다. 이것은 TPU 활용률을 개선하기 위해 수행하게 될 성능 엔지니어링 종류의 좋은 예입니다. 더 많은 `shard_map` 예제를 보려면 [이 노트가 훌륭합니다](https://jax.readthedocs.io/en/latest/notebooks/shard_map.html#example-1-all-gather-on-one-side).

이제 `jax.jit` 또는 `shard_map`을 사용하여 구현해 볼 수 있는 유용한 연습 문제 몇 가지를 소개합니다!

## Worked Problems

다음은 무작위 JAX 관련 문제입니다. 나중에 더 추가할 예정입니다. 이 모든 것을 위해서는 Colab에 일정 수의 TPU가 필요합니다. TPUv2-8이 있는 공개 Colab을 사용할 수 있습니다. 지금부터는 N개의 디바이스를 사용할 수 있다고 가정합니다.

**Problem 1:** **A**가 `X * Y = N`인 float32[S<sub>X</sub>, D<sub>Y</sub>] 형태의 활성화 배열이라고 합시다. 다음을 수행하세요:

1. 각 `(X, Y)` 샤드 내에서 평균을 계산하는 JAX 함수를 작성하세요. 즉, `arr[i, j]`가 샤드 `(i, j)`에 대한 평균인 [X, Y] 크기의 배열을 반환합니다. `jax.jit`과 `shard_map` 모두로 이 작업을 수행하세요. 각각 프로파일링하여 얼마나 걸렸는지 확인하세요. 통신이 추가되었나요? *힌트: 없어야 하지만 때때로 XLA가 어쨌든 추가합니다.*

2. **각 샤드 X 내에서** 일부 shift에 대해 roll(x, shift, axis=0) - x를 반환하는 JAX 함수를 작성하세요. jax.jit으로 이것을 하라고 할 만큼 가학적이지 않으니 그냥 `shard_map`으로 하세요.

{% details 답을 보려면 여기를 클릭하세요. %}

Part 1: 파트 1에 대한 해결책은 다음과 같습니다. `jax.jit` 솔루션을 위해 수행해야 하는 꽤 복잡한 reshape에 유의하세요.

```python
import numpy as np

import jax
import jax.numpy as jnp

mesh = jax.make_mesh((4, 2), ('X','Y'))

average_shmap = jax.shard_map(
    lambda x: x.mean(keepdims=True),
    mesh=mesh,
    in_specs=jax.P('X','Y'), out_specs=jax.P('X','Y')
average_shmap = jax.shard_map(
    lambda x: x.mean(keepdims=True),
    mesh=mesh,
    in_specs=jax.P('X','Y'), out_specs=jax.P('X','Y')
)

def average(x):
  X, Y = mesh.axis_sizes
  return x.reshape(X, x.shape[0] // X, Y, x.shape[1] // Y).mean(axis=(1, 3))

average_jit = jax.jit(average, out_shardings=jax.NamedSharding(mesh, jax.P('X','Y')))
average_jit = jax.jit(average, out_shardings=jax.NamedSharding(mesh, jax.P('X','Y')))

x = jnp.arange(8 * 64 * 8, dtype=jnp.int32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, jax.P('X','Y')))
x = jax.device_put(x, jax.NamedSharding(mesh, jax.P('X','Y')))

y1 = average_shmap(x)
y2 = average_jit(x)

np.testing.assert_array_equal(y1, y2)
```

Part 2: 파트 2에 대한 유사한 해결책은 다음과 같습니다.

```python
import numpy as np

import jax
import jax.numpy as jnp

import functools

P = jax.sharding.PartitionSpec

mesh = jax.make_mesh((4, 2), ('X','Y'))

def shift_shmap(x, shift: int):
  shmapped = jax.shard_map(
      lambda x: jnp.roll(x, shift, axis=0),
      mesh=mesh,
      in_specs=jax.P('X','Y'), out_specs=jax.P('X','Y')
  shmapped = jax.shard_map(
      lambda x: jnp.roll(x, shift, axis=0),
      mesh=mesh,
      in_specs=jax.P('X','Y'), out_specs=jax.P('X','Y')
  )
  return shmapped(x)

@functools.partial(jax.jit, static_argnames=['shift'], out_shardings=jax.NamedSharding(mesh, jax.P('X','Y')))
@functools.partial(jax.jit, static_argnames=['shift'], out_shardings=jax.NamedSharding(mesh, jax.P('X','Y')))
def shift_jit(x, shift: int):
  X, Y = mesh.axis_sizes
  reshaped = x.reshape(X, x.shape[0] // X, -1)
  return jnp.roll(reshaped, shift, axis=1).reshape(x.shape[0], x.shape[1])

x = jnp.arange(8 * 64 * 8, dtype=jnp.int32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, jax.P('X','Y')))
x = jax.device_put(x, jax.NamedSharding(mesh, jax.P('X','Y')))

y1 = shift_shmap(x, 5)
y2 = shift_jit(x, 5)

np.testing.assert_array_equal(y1, y2)
```

{% enddetails %}

**Problem 2:** 여기서는 기본 "mixture of experts" 모델을 함께 만들어 보겠습니다. **W**: float32[E<sub>X</sub>, D, F]를 E개의 "expert" 행렬 세트라고 합시다. **A**: float32[S<sub>X</sub>, D] (우리의 활성화)와 **B**: int32[S<sub>X</sub>]를 "라우팅 할당" 세트라고 합시다. 여기서 B[i]는 해당 활성화를 처리할 행렬을 알려주는 `[0, E)` 범위의 정수입니다. `Out[i] = W[B[i]] @ A[i]`를 반환하는 JAX 함수를 작성하고 싶습니다.

1. 샤딩을 완전히 무시하는 것으로 시작합시다. 이 텐서들을 모두 한 디바이스에 맞을 만큼 작게 만드세요. 이 함수의 로컬 구현을 작성하세요. *`[S, D, F]` 형태의 배열을 구체화하지 마세요! 힌트: 마스킹에 주의하면서 토큰을 `[E, S, D]` 형태의 새 버퍼로 정렬해 보세요 (왜 두 번째 차원의 크기가 S여야 할까요?).*

2. 위의 메서드를 그냥 `jax.jit`하면 어떤 일이 일어날 것입니다. 이것을 프로파일링하고 어떤 통신을 하기로 결정했는지 확인하세요. 얼마나 걸리나요?

3. 위에서 눈치챘을 한 가지 문제는 전체 활성화 세트 **A**를 로컬로 gather할 가능성이 높다는 것입니다. 즉, AllGather<sub>X</sub>([S<sub>X</sub>, D]). 이는 통신 면에서 비쌀 뿐만 아니라 전체 활성화 세트를 로컬에 맞출 수 없다면 메모리 면에서도 엄청나게 비쌉니다. `shard_map`과 명시적 통신을 사용하여 위를 구현하세요.

      1. 첫 번째 패스의 경우 `jax.lax.all_gather`를 사용하고 (a)처럼 재정렬하는 것이 가장 쉬울 수 있습니다.

      2. 두 번째 패스의 경우 `[E, S, D]` 크기의 배열을 구체화하지 않도록 해보세요. 즉, `jax.lax.while_loop` 내부에서 `jax.lax.all_to_all`을 사용하여 울퉁불퉁한(ragged) 방식으로 계산을 수행해 보세요. 이렇게 하면 전체 활성화를 구체화하고 패딩에 컴퓨팅을 낭비하는 것을 피할 수 있습니다. 원래 구현보다 얼마나 빠른가요?

4. 대부분의 MoE는 여러 (k) 전문가에게 라우팅한 다음 결과를 평균화합니다. 이것을 구현하기 위해 위를 리팩터링하세요. 이 경우 k개의 전문가에게 라우팅하기 위해 **B**: int32[S, k]라고 합시다.

{% details 답을 보려면 여기를 클릭하세요 (일부). %}

1/2. 파트 (1)의 경우 많은 선택지가 있습니다. 다음은 마스킹을 사용하여 전문가를 반복하는 한 가지 옵션입니다.

```python
def moe_local(W: jnp.ndarray, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    S, _ = A.shape
    E, _, F = W.shape

    def expert_forward(carry, e):
        output = carry  # [S, F]
        mask = (B == e)[:, None]  # [S, 1]
        expert_result = A @ W[e]  # [S, F] - this expert's transform of ALL tokens
        output = output + expert_result * mask  # Only keep results for assigned tokens
        return output, None

    output = jnp.zeros((S, F))
    output, _ = lax.scan(expert_forward, output, jnp.arange(E))

    return output
```

`jax.lax.ragged_dot`을 사용할 수도 있는데, 이는 비슷하지만 더 효율적으로 수행합니다.

3. 여기서는 의사 코드만 스케치하겠습니다(깔끔한 솔루션이 있다면 자유롭게 추가해 주세요):

```python
chunk_size = 128
def matmul(W, x, B):
  i = 0
  x = # sort x according to assignments
  while (chunk := x[i:i+chunk_size].any()):
     chunk = all_to_all(chunk)
     out = matmul_local(W, chunk)
  return concat(out)
```

기본 아이디어는 배열의 청크를 반복하고, 정렬하고 all_to_all을 수행한 다음, 로컬 FLOPs를 수행하는 것입니다.

{% enddetails %}

**Problem 3:** 위의 collective matmul 예제는 실제 LLM에 매우 관련이 있습니다. 전체 Transformer 스택을 수행하도록 예제를 조정해 봅시다.

1. 연습 삼아 AllReduce collective matmul, 즉 A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F] -> Out[B<sub>X</sub>, F]를 구현하는 것부터 시작해 봅시다. 출력은 복제되지 않는다는 점에 유의하세요. 순진한 알고리즘은 위에서 논의되었으며, 기본적으로 로컬 matmul 다음에 AllReduce가 옵니다. 이 작업의 통신 중첩 "collective" 버전을 만들어 보세요. *힌트: 출력 차원에 대해 타일링하고 `jax.lax.psum`(일명 AllReduce)을 자유롭게 사용하세요.* *참고: XLA가 이를 처리하는 방식 때문에 실제로는 기준선보다 빠르지 않을 수 있습니다.*

2. 위의 AllReduce collective matmul에 대한 보완은 Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W2[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>]와 같은 ReduceScatter collective matmul입니다. 이는 Transformer의 down-projection 행렬에서 발생합니다. JAX에서 이의 collective, 중첩 버전을 구현하세요. 필요한 최소한의 데이터만 전달하도록 주의하세요. *힌트: 누적할 때 결과를 순열(permute)해 보세요.*

3. 이 두 가지를 합쳐서 In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>]를 중첩된 통신으로 수행하는 엔드투엔드 Transformer 블록을 만드세요.<d-footnote>이전과 마찬가지로, 여기서는 생략한 비선형성 때문에 $W_{in} \cdot W_{out}$을 먼저 수행할 수 없습니다.</d-footnote> `jax.jit` 구현보다 얼마나 빠른가요?

**Problem 4:** 위에서 구현된 모든 collective matmul은 단방향입니다. 즉, 한 방향으로만 순열합니다. collective AllReduce matmul과 collective ReduceScatter matmul을 양방향 통신을 사용하도록 다시 작성하세요. 이것들은 얼마나 더 빠른가요?

### 파트 10은 여기까지입니다. 기본적으로 이게 전부입니다! 최종 결론과 추가 자료를 보려면 [여기](../conclusion)를 클릭하세요.