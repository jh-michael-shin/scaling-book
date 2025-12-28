---
layout: distill
title: "Conclusions and Further Reading"
# permalink: /main/
description: "읽어주셔서 감사합니다! 여기에 추가 학습을 위한 몇 가지 참조 자료를 포함하겠습니다."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 11

previous_section_url: "../jax-stuff"
previous_section_name: "Part 10: JAX"

next_section_url: "../gpus"
next_section_name: "Part 12: GPUs"
next_section_url: "../gpus"
next_section_name: "Part 12: GPUs"

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
  - name: "Acknowledgments"
  - name: "Further Reading"
  - name: "Feedback"

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
  .algorithm {
    padding: 10px;
    margin-top: 5px;
    margin-bottom: 5px;
    border-style: dashed;
    background-color: #fffaf2;
  }

  .algorithm li {
    margin-bottom: 0px;
  }
---

<p markdown=1 class="takeaway">
<b>번역 안내:</b> 원저자([Jacob Austin](https://www.jacobaustin.org/))의 허락을 받아 원문을 번역 중입니다.<br> 
해당 글의 1인칭은 원문 저자를 지칭합니다.<br> 
원문: [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)<br> 
번역: [신종훈](https://www.linkedin.com/in/michael-shin-3522a6189/)</p>

**이 에세이 시리즈를 읽어주셔서 감사하며 끝까지 함께해주신 것을 축하드립니다.** 결론을 내리기 전에 몇 가지 감사의 말씀을 드립니다:

## Acknowledgments

이 문서는 Google DeepMind의 많은 분들의 상당한 집단적 투자를 나타내며, 이분들께 감사를 표하고 싶습니다!

* James Bradbury, Reiner Pope, Blake Hechtman은 이 원고의 많은 아이디어를 처음 도출했으며, Transformer의 시스템적 관점을 일찍부터 이해하고 있었습니다.
* Sholto Douglas는 이 문서의 첫 번째 버전을 작성했으며 프로젝트를 시작하는 데 책임을 맡았습니다. 그는 누구보다 이 문서의 전반적인 서사에 대한 책임이 있습니다.
* Jacob Austin은 거친 메모에서 더 다듬어지고 포괄적인 결과물로 변환하는 작업을 주도했습니다. 그는 편집, 서식 지정 및 이 문서의 배포 작업의 많은 부분을 수행했으며 다른 저자의 기여를 조정했습니다.
* 대부분의 그림과 애니메이션은 Anselm Levskaya와 Charlie Chen이 만들었습니다.
* Charlie Chen은 추론 섹션을 작성하고 많은 추론 그림을 그렸습니다.
* Roy Frostig는 출판, 편집 및 여정의 여러 단계에서 도움을 주었습니다.

또한 프로세스 전반에 걸쳐 비판적인 피드백을 주신 많은 분들, 특히 Zak Stone, Nikhil Sethi, Caitlin Stanton, Alek Dimitriev, Sridhar Lakshmanamurthy, Albert Magyar, Diwakar Gupta, Jeff Dean, Corry Wang, Matt Johnson, Peter Hawkins 외 많은 분들께 감사드립니다. HTML 서식 지정에 도움을 준 Ruiqi Gao에게도 감사드립니다.

**모두 감사합니다!**

<p markdown=1 class="announce">가시기 전에 NVIDIA GPU에 관한 새로운 [12장](../gpus)도 읽어보시면 좋을 것 같습니다!</p>

## Further Reading

관련된 글들이 많이 있습니다:

* [**TPU Deep Dive**](https://henryhmko.github.io/posts/tpu/tpu.html): 이 책의 정신에 부합하는 TPU 아키텍처에 대한 훌륭한 심층 분석입니다.
* [**Domain specific architectures for AI inference**](https://fleetwood.dev/posts/domain-specific-architectures): 이 책의 정신에 부합하는 하드웨어 및 모델 심층 분석입니다.
* [**A Domain-Specific Supercomputer for Training Deep Neural Networks**](https://dl.acm.org/doi/pdf/10.1145/3360307): OG TPU 논문 중 하나로, 여기서는 다루지 않은 Google TPU 프로그램에 대한 훌륭한 세부 정보가 많이 포함되어 있습니다.
* [**Making Deep Learning Go Brrrr From First Principles**](https://horace.io/brrr_intro.html): LLM 루프라인 및 성능 엔지니어링에 대한 더 GPU 및 PyTorch 중심의 튜토리얼입니다.
* [**Writing TPU Kernels with Pallas**](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html): TPU 프로그래밍은 점점 더 Pallas에서 커스텀 커널을 작성하는 것을 포함하고 있습니다. 이 시리즈는 커널 작성 방법과 여기서 언급되지 않은 많은 하위 수준 TPU 세부 정보를 다룹니다.
* [**How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog**](https://siboehm.com/articles/22/CUDA-MMM): GPU 및 CUDA 전용이지만, CUDA에서 matmul 커널을 최적화하는 방법을 보여주는 훌륭한 블로그 게시물입니다. TPU와 GPU가 어떻게 다른지 심층 분석하기에 좋을 수 있습니다.
* [**Distributed arrays and automatic parallelization**](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html): JAX의 병렬 처리 API에 대한 정말 좋은 가이드이며, 우리가 여기서 논의한 아이디어 중 일부를 실제로 구현하는 방법을 배우기에 좋은 방법입니다.
* [**Rafi Witten's High Performance LLMs 2024 Class**](https://github.com/rwitten/HighPerfLLMs2024): 전 동료인 Rafi가 TPU 성능 엔지니어링에 대한 훌륭한 강의를 했으며 슬라이드는 모두 GitHub에 있습니다. 여기서는 우리가 다루는 것보다 더 깊이 있게 많은 것을 다룹니다.
* [**\[2211.05102\] Efficiently Scaling Transformer Inference**](https://arxiv.org/abs/2211.05102): Transformer 추론의 수학에 대한 상세한 논문입니다. 이것은 이 문서의 많은 부분에 영감을 주었습니다.
* [**Huggingface Ultra-Scale Playbook**](https://huggingface.co/spaces/nanotron/ultrascale-playbook): 이 책의 GPU 버전과 같은 것으로, 훈련 중 PyTorch가 병렬 처리 기술 및 메모리 절약 기술을 구현하는 방법에 대해 더 깊이 이야기합니다.
* [**Transformer Inference Arithmetic**](https://kipp.ly/transformer-inference-arithmetic/): 이 책과 동일한 아이디어와 훌륭한 일러스트레이션이 많이 포함된 블로그입니다.
* [**Stanford CS336 Slides and Videos**](https://stanford-cs336.github.io/spring2025/index.html#coursework): 유용한 연습 문제와 함께 LLM 훈련 및 서빙의 많은 세부 정보를 다루는 환상적인 스탠포드 과정입니다. 과제 1과 2가 특히 관련이 있습니다.
* [**Stas Bekman's ML Engineering Handbook**](https://github.com/stas00/ml-engineering): 클라우드 공급자와의 협상 방법, 클러스터 관리, GPU 처리량의 실증적 측정과 같이 이 책에서 다루지 않은 주제를 다루는 ML 인프라에 대한 매우 실용적인 가이드입니다.

이 분야에는 포괄적인 글을 쓸 여지가 여전히 많으므로, 이 원고가 더 많은 글을 장려하기를 바랍니다! 또한 우리는 이것이 연구하고 조사하기에 유익한 분야라고 믿습니다. 많은 경우 하드웨어 가속기가 많이 없어도 수행할 수 있습니다.

## Feedback

더 개선할 수 있도록 의견이나 질문을 남겨주세요. 교신 저자인 Jacob Austin에게 jacobaustin123 [at] gmail [dot] com으로 연락하거나 [GitHub](https://github.com/jax-ml/scaling-book)에 문제, 풀 리퀘스트 또는 토론을 게시하여 편집을 제안할 수 있습니다.