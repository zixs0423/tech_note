---
layout: default
---


- [Deployment](#deployment)
  - [Repository](#repository)
    - [Git](#git)
      - [Concepts](#concepts)
      - [Source](#source)
      - [Code](#code)
  - [Environment](#environment)
    - [Anaconda](#anaconda)
      - [Concepts](#concepts-1)
      - [Source](#source-1)
      - [Code](#code-1)
    - [Pip](#pip)
      - [Concepts](#concepts-2)
      - [Source](#source-2)
      - [Code](#code-2)
    - [uv](#uv)
      - [Concepts](#concepts-3)
      - [Source](#source-3)
      - [Code](#code-3)
    - [Nvidia](#nvidia)
      - [Concepts](#concepts-4)
      - [Source](#source-4)
      - [Code](#code-4)
  - [Packages](#packages)
    - [Python](#python)
      - [Concepts](#concepts-5)
      - [Source](#source-5)
      - [Code](#code-5)
    - [Torch](#torch)
      - [Concepts](#concepts-6)
      - [Source](#source-6)
      - [Code](#code-6)
    - [Transformers](#transformers)
      - [Concepts](#concepts-7)
      - [Source](#source-7)
      - [Code](#code-7)


# Deployment

## Repository

### Git

#### Concepts

* git clone 'remote repository web URL'
  
  clone the remote repository (only the default branch, main/master)

* git branch 
  
  get the list of local branches

* git branch -r
  
  get the list of remote branches

* git checkout -b 'new-local-branch-name' origin/'remote-branch-name'
  
  pull a new remote branch

* git checkout 'local-branch-name'
  
  switch to a branch already existed locally

* git pull 
  
  fetch and merge changes from the remote branch that your current local branch is tracking

* git checkout -b 'new-local-branch-name'
  
  create a new local branch

* git branch -d 'local-branch-name'
  
  delete the local branch

* git push origin --delete 'remote-branch-name'
  
  delete the remote branch

* git add .
  
  add all modifications from working directory to staging area

* git commit -m ""
  
  commit with message ""

* git push origin
  
  push your current branch to the remote

* git status
  
  get the current branch and commits

* create a .gitignore file in the project folder. Write '*.csv' in it to ignore all the csv files.

<br>

#### Source

[Geeksforgeeks Git Tutorial](https://www.geeksforgeeks.org/git/git-tutorial/)

[Understanding Git-Ignore and Its Usage](https://www.geeksforgeeks.org/git/what-is-git-ignore-and-how-to-use-it/)

[Geeksforgeeks Naming Conventions for Git Branches](https://www.geeksforgeeks.org/git/how-to-naming-conventions-for-git-branches/)

<br>

#### Code

* merge a feature branch to main branch manually
  * git pull: This fetches changes from the remote and merges them into your current branch
  * git checkout main: You switch to the main branch.
  * git merge feature: This merges the local feature branch into your main branch.
  * git push: Finally, you push your local main branch to the remote.
* Pull Request (PR): a request to **merge** one branch (usually a feature branch) into another branch (usually main or master) in a remote repository.
* copy a github repository to the team project repository
  * cd 'local code folder'
  * git clone: clone the github repository to local environment
  * git clone: clone the team project repository to local environment
  * cd 'local team project repository folder'
  * (git checkout -b 'new-local-branch-name': create a new local branch if needed)
  * copy all the files in the local github repository to the local team project repository manually
  * git add .
  * git commit -m ""
  * git push origin
  
<br>

---

## Environment

### Anaconda

#### Concepts

* conda create -n 'env_name' python=='python_version'
  
  create a new environment

* conda activate 'env_name'

* conda info --envs
  
  list all environments and locations
 
* conda list
  
  list installed packages

* conda remove --name 'env_name' --all
  
  delete environment by name

* conda config --show-sources
  
  view channel sources

* conda config --remove-key channels
  
  remove all channel sources

* conda config --remove-key channels
  
  remove all channel sources

* conda config --add channels 'channel_name'
  
  add the aliyun channel sources:
  
  conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/main
  
  conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/r
  
  conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/msys2

  other sources:

  mirrors.ustc.edu.cn

  mirrors.bfsu.edu.cn

  mirrors.tuna.tsinghua.edu.cn

* The structure:
  
  ~/anaconda3/envs/my_env/bin/python  the currently active Python interpreter
  
  ~/anaconda3/envs/my_env/lib/python3.x/site-packages/  pip downloads the wheels and extracts the package files directly into this path
  
* Conda "owns" the environment, but pip populated it. Virtual environments are designed to be isolated, so the package downloaded by pip in one environment can not be used in another environment.


<br>

#### Source

[Cheatsheet](https://docs.conda.io/projects/conda/en/stable/user-guide/cheatsheet.html)

[Commands](https://docs.conda.io/projects/conda/en/stable/commands/index.html)

<br>

#### Code

<br>

---

### Pip

#### Concepts

* pip install -e .[extra]
  
  installing the current project as a package in editable mode of pip.
  
  extra could be torch or some other packages need to be installed.
  
  If you ran uv pip install -e ., a tiny link is placed in the Site-Packages folder that points back to your Project Folder.

* pyproject.toml : 
  
  the modern standard for configuring Python projects, designed to replace the older, fragmented system of using multiple files like setup.py, setup.cfg, requirements.txt, and MANIFEST.in.
  
  When you ran pip install -e .[torch], pip looked into the [project.optional-dependencies] section of your pyproject.toml, found the list of packages associated with torch, and installed them along with your local code.

<br>

#### Source

<br>

#### Code

* create a environment to fullfill the requirement of a github repository
  * conda create -n 'env_name' python=='python_version'
  * conda activate 'env_name'
  * pip install -r requirements.txt

<br>

---

### uv

#### Concepts

* a single tool that replaces pip, pip-tools, venv, pyenv, and poetry all at once. uv is often 10x to 100x faster than pip
  
* uv venv: 
  
  Create a virtual environment
  
* source .venv/bin/activate:
  
  Activate the environment
  
* uv pip install : 
  
  install the package

<br>

#### Source

<br>

#### Code

<br>

---

### Nvidia

#### Concepts

* nvidia-smi
  
  Shows a summary table with:
  
  GPU index, name, and UUID
  
  Driver & CUDA versions

  GPU & memory utilization

  Power consumption and temperature

  Active processes using the GPU

* watch -n 1 nvidia-smi
  
  looping

<br>

#### Source

<br>

#### Code

<br>

---

## Packages

### Python

#### Concepts

<br>

#### Source

<br>

#### Code

<br>

---

### Torch

#### Concepts

* pip install torch=='torch_version'+cu'cuda_version'
  the cuda_version here should be smaller than the cuda version got by 'nvidia-smi'

<br>

#### Source

[pytorch](https://pytorch.org/get-started/locally/)

<br>

#### Code

<br>

---

### Transformers

#### Concepts

<br>

#### Source

<br>

#### Code

* Download and save LLMs on Hugging Face
  * pip install transformers
  * pip install torch
  * run this python script
  
    ```python
    from transformers import LlamaModel, LlamaTokenizer
    model = LlamaModel.from_pretrained('huggyllama/llama-7b')
    tokenizer = LlamaTokenizer.from_pretrained('huggyllama/llama-7b')
    model.save_pretrained('./llama-7b')
    tokenizer.save_pretrained('./llama-7b')
    ```

  * next time you can load from local folder
  
    ```python
    from transformers import LlamaModel, LlamaTokenizer
    model = LlamaModel.from_pretrained('./llama-7b')
    tokenizer = LlamaTokenizer.from_pretrained('./llama-7b')
    ```
  
<br>

---