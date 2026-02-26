---
layout: default
---


- [Workflow](#workflow)
  - [Repository](#repository)
    - [Git](#git)
      - [Concepts](#concepts)
      - [Source](#source)
      - [Practice](#practice)
  - [Environment](#environment)
    - [Anaconda](#anaconda)
      - [Concepts](#concepts-1)
      - [Source](#source-1)
      - [Practice](#practice-1)


# Workflow

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

<br>

#### Source

[Geeksforgeeks Git Tutorial](https://www.geeksforgeeks.org/git/git-tutorial/)

[Geeksforgeeks Naming Conventions for Git Branches](https://www.geeksforgeeks.org/git/how-to-naming-conventions-for-git-branches/)

<br>

#### Practice

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

<br>

#### Source

<br>

#### Practice

<br>

---