# RAG-Privacy: Evaluating and Mitigating Information Leakage in Retrieval-Augmented Generation Systems

## Overview

This repository implements a complete pipeline for **privacy evaluation**, **leakage analysis**, and **defense mechanisms** for Retrieval-Augmented Generation (RAG) systems.
It builds on the findings of the ACL 2024 paper:

**"The Good and The Bad: Exploring Privacy Issues in Retrieval-Augmented Generation (RAG)"
Zeng et al., ACL 2024.**

and integrates an auditing module inspired by:

**"S-RAG: Auditing Data Provenance in Retrieval-Augmented Generation Systems"
Zeng et al., ACL 2025.**

The pipeline supports:

* Regex-based PII sanitization (emails, phones, URLs, simple addresses)
* Context summarization to reduce leakage
* Prompt based mitigation strategies
* Privacy-utility trade-off measurement
* S-RAG-style membership inference auditing
* Extraction of qualitative leakage examples

The goal is to quantify and mitigate both **retrieval-stage leakage** and **generation-stage leakage** in RAG systems.

---

## System Architecture

```
Dataset
   |
   v
Retrieval DB --> Sanitization / Summarization / Prompt-Engineering --> LLM --> Evaluation
    |                                                                   ^
    v                                                                   |
    +-------------------------------------------------------------------+
                                 |
                        	S-RAG Audit
```

---

## Features at a Glance

* End-to-end RAG privacy benchmarking
* Leakage metrics for both retriever and generator
* Pluggable sanitizers and summarizers
* Reproducible S-RAG audit pipeline
* Modular scripts for prompt generation, retrieval DB creation, and evaluation

---

## Setup Instructions

Install project dependencies:

```bash
conda env create -f environment.yml
conda activate raglm
```

We are using llama-2-7b-chat as the base LLM and bge-large-en-v1.5 as the embedding model.
Follow [Llama 2 install guide](https://github.com/facebookresearch/llama?tab=readme-ov-file#quick-start) to install Llama model. **Note** that you should download the models (including the file `tokenizer.model` and the folder that store the parameters of the Llama) in folder  `.\Model`. Similarly, download the embedding model from [BGE model page](https://huggingface.co/BAAI/bge-large-en-v1.5) and put it in folder `.\Model`.

To ensure deterministic results, set the random seed in `src/seed_utils.py`. Run this file using the following command:

```bash
python src/seed_utils.py
```

---

## Repository Structure

```
CS787-Course-Project/
  Data/
    chatdoctor/
      chatdoctor.txt
  Model/
    tokenizer.model
    llama-2-7b-chat/
    bge-large-en-v1.5/
  llama/
  Information/
  S-RAG_analysis/
  README.md
  environment.yml
  src/
    retrieval_database.py
    evaluation_baseline.py
    evaluation_results.py
    generate_prompt.py
    generate_prompt_sanitize.py
    generate_prompt_instruct.py
    generate_prompt_summarizer.py
    run_language_model.py
    run_language_model_summarizer.py
    llm_as_judge.py
    seed_utils.py
    master.sh
```

Key directories:

* `Data/`: evaluation and audit datasets
* `Model/`: base LLM, embedding model and tokenizer
* `src/`: all RAG, sanitization, summarization, evaluation, and prompt generation scripts
* `S-RAG_analysis/`: auditing module
* `Information/`: Information useful for attacks

---

## About the Codebase

To run the complete RAG privacy evaluation and mitigation pipeline, follow these steps:
```bash
cd src
```
### 1. Retrieval Database Construction
Build embeddings and store a persistent vector DB in a directory `../RetrievalBase` in the  of the chatdoctor dataset. To create the retrieval database, run:
```bash
python retrieval_database.py --dataset_name "chatdoctor-train" -encoder_model "bge-large-en-v1.5"
```
Note: Before running `retrieval_database.py` ensure `Data` contains only `chatdoctor` directory. Also ensure that there is no `RetrievalBase` directory before running this script

### 2. Generate prompts for RAG model
Generate prompts using the information from `Information` folder and the retrieved contexts from the retrieval database using questions. This will create a script (`chat-target-email.sh`) which gives these prompts to LLM and gets its response. In addition to this it creates a folder `Inputs&Outputs` which would have the prompts and contexts stored as .json. To generate prompts, run:
```bash
python generate_prompt.py
```

Variants:

* `generate_prompt_instruct.py` - for instruction-tuned mitigation prompts
* `generate_prompt_sanitize.py` - for generating prompts with sanitized contexts
* `generate_prompt_summarizer.py` - for generating prompts with summarized contexts

In order to capture the privacy utility tradeoff we define a baseline task involving comparision of generated responses by the RAG pipeline with chatdoctor responses for a small subset of the `chatdoctor.txt` dataset. We can run to compute results of baseline task as follows:
- Run `generate_prompt.py` (or its variants) with  `--flag=1` while running the above generate_prompt.py or their variants. An example is as follows:
```bash
python generate_prompt.py --flag=1
``` 


### 3. Run the LLM
Run the script generated in the above process to run the LLM with the generated prompts (for both attack and baseline cases). The generated outputs are stored in the `Inputs&Outputs` directory as .json
Example:

```bash
sh ./chat-target-email.sh
```

### 4. Compute evaluation metrics
The final step is to compute the evaluation metrics for the RAG model outputs. This includes both privacy leakage metrics and utility metrics. To evaluate attack results, run:

```bash
python evaluation_results.py --exp_name="chat-target-email"
```

To run evaluate accuracy for baseline tasks, use the following command :
```bash
python evaluation_baseline.py --exp_name="chat-target-email"
```

### 5. S-RAG Membership Inference Auditing

To run the S-RAG membership inference auditing module, navigate to the `S-RAG_analysis` directory and follow the instructions in the `README.md` file located there.

### 6. LLM as a judge

In this script we try to use the LLM to produce an accuracy for the baseline task. It takes 3 files, the gold response file from the dataset, the output for baseline task without any mitigation and the output for baseline task with the mitigation method and compares which of them is closer to the golden response. Output `0` if 1st is better `1` if its a tie `2` if 2nd is better and `NO_FLAG` if response doesn't contain the verdict.

```bash
python llm_as_judge.py <Model_name> <path_dataset> <path_normal> <path_mitigate> <path_settings>
```

### 7. Master run script

Master script can perform all the mitigation/baseline tasks and complete the evaluation

```bash
sh ./master.sh attack normal chat-target-email
```

---

## Citation

**RAG Privacy Paper (ACL 2024)**
Zeng, Shenglai et al.
"The Good and The Bad: Exploring Privacy Issues in Retrieval-Augmented Generation (RAG)."
ACL 2024.
[https://arxiv.org/abs/2402.16893](https://arxiv.org/abs/2402.16893)

**S-RAG Paper (ACL 2025)**
Zeng, Zhiheng et al.
"S-RAG: Auditing Data Provenance in Retrieval-Augmented Generation Systems."
ACL 2025.
[https://aclanthology.org/2025.acl-long.512.pdf](https://aclanthology.org/2025.acl-long.512.pdf)

---

## Contributors

This repository is maintained as part of **CS787 Project, IIT Kanpur**.

* Akanksha Wattamwar (221214)
* Mahaarajan J (220600)
* Animesh Madaan (220145)
* Pahal Patel (220742)