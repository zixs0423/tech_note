---
layout: default
---

- [LLM](#llm)
  - [Reinforcement Learning](#reinforcement-learning)
    - [Q-learning](#q-learning)
      - [Concepts](#concepts)
      - [Source](#source)
      - [Code](#code)
    - [RLHF](#rlhf)
      - [Concepts](#concepts-1)
      - [Source](#source-1)
      - [Code](#code-1)
  - [Local Deployment](#local-deployment)
    - [Ollama](#ollama)
      - [Concepts](#concepts-2)
      - [Source](#source-2)
      - [Code](#code-2)
    - [vLLM](#vllm)
      - [Concepts](#concepts-3)
      - [Source](#source-3)
      - [Code](#code-3)


# LLM

## Reinforcement Learning

### Q-learning

#### Concepts
<br>

#### Source

[Geeksforgeeks Reinforcement Learning](https://www.geeksforgeeks.org/machine-learning/what-is-reinforcement-learning/)

<br>

#### Code

[RL_maze](../code/RL_maze.py)

<br>

---

### RLHF

#### Concepts

1. RLHF
2. PPO
3. DPO

<br>

#### Source

[Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study](https://arxiv.org/pdf/2404.10719)

ICML 2026 Cited by 244

[Direct Preference Optimization:Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290#page=4.22)

NeurIPS 2023 Cited by 7219


<br>

#### Code

<br>

---

## Local Deployment

### Ollama

#### Concepts

* Download the ollama on https://ollama.com/.
* Open a terminal and input 'ollama --version' to check if the downloading is successful.
* Download the LLMs by inputing 'ollama pull llama3'
* Open a teminal and input: 
  ```python
  curl http://localhost:11434/api/generate -d '{
  "model": "llama3",
  "prompt": "Why is the sky blue?"}'
  ```
  to try to interact with the LLMs.
* There is also a UI interface in ollama in which users can talk with the LLMs, including the pulled ones, in a breifer way.


<br>

#### Source

[Ollama API](https://docs.ollama.com/api/introduction)

[Run LLMs Locally: 6 Simple Methods](https://www.datacamp.com/tutorial/run-llms-locally-tutorial)

[VS Code Integration](https://docs.ollama.com/integrations/vscode)

<br>

#### Code

<br>

---

### vLLM

#### Concepts

<br>

#### Source

[Performance vs Practicality: A Comparison of vLLM and Ollama](https://robert-mcdermott.medium.com/performance-vs-practicality-a-comparison-of-vllm-and-ollama-104acad250fd)

<br>

#### Code

<br>

---