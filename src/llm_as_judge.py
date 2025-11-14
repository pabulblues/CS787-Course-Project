#!/usr/bin/env python3
"""
llm_as_judge.py

LLM-as-a-Judge script comparing baseline vs guardrail outputs
against gold ChatDoctor answers.

Labels:
###0 -> Without guardrails closer
###1 -> Both equally close
###2 -> With guardrails closer
NO_FLAG_FOUND -> Judge did not produce a valid label

ASCII-only version.
"""

import os
import re
import json
import warnings
import fire
from llama import Llama
from langchain_openai import OpenAI


# ===========================================================
# Load ChatDoctor Gold
# ===========================================================
def load_chatdoctor_test(file_path):
    print("Loading ChatDoctor gold file:", file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read().strip().split("\n\n")

    questions, answers = [], []
    for item in data:
        input_match = re.search(r"input:\s*(.*)", item, re.DOTALL)
        output_match = re.search(r"output:\s*(.*)", item, re.DOTALL)
        if input_match and output_match:
            questions.append(input_match.group(1).strip())
            answers.append(output_match.group(1).strip())

    print("Loaded", len(answers), "gold QA pairs.")
    return questions, answers


# ===========================================================
# Load Generated Outputs
# ===========================================================
def load_json_outputs(path):
    print("Loading generated outputs from:", path)
    with open(path, "r", encoding="utf-8") as f:
        outputs = json.load(f)
    print("Loaded", len(outputs), "generated responses.")
    return outputs


# ===========================================================
# MAIN
# ===========================================================
def main(
        ckpt_dir: str,
        gold_file: str,
        without_guardrails_path: str,
        with_guardrails_path: str,
        settings_path: str,
        tokenizer_path: str = "tokenizer.model",
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_seq_len: int = 4096,
        max_gen_len: int = 16,     # STRICT: enough for ###0
        max_batch_size: int = 1
):
    print("LLM-as-a-Judge Evaluation Started")
    print("---------------------------------")

    # =====================================================
    # Stage 0: Distributed settings
    # =====================================================
    print("Stage 0: Loading distributed settings...")

    dist_defaults = {
        "RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "12345"
    }
    dist_cfg = dist_defaults.copy()

    if settings_path and os.path.exists(settings_path):
        print("Found settings.json:", settings_path)
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                s = json.load(f)
            if "distributed" in s:
                for k, v in dist_defaults.items():
                    dist_cfg[k] = str(s["distributed"].get(k, v))
        except Exception as e:
            print("Error loading settings.json:", e)

    print("Applying distributed configuration:")
    for k, v in dist_cfg.items():
        os.environ[k] = v
        print("  ", k, "=", v)

    # =====================================================
    # Stage 1: Load model
    # =====================================================
    print("Stage 1: Loading model:", ckpt_dir)

    if "gpt" in ckpt_dir:
        if ckpt_dir == "gpt":
            ckpt_dir = "gpt-3.5-turbo-instruct"
        llm = OpenAI(model=ckpt_dir, temperature=temperature,
                     top_p=top_p, max_tokens=max_gen_len)
        generator = None
        flag_llm = "gpt"
        print("Using OpenAI GPT model:", ckpt_dir)

    else:
        model_path = "../Model/" + ckpt_dir
        print("Loading local LLaMA checkpoint from:", model_path)

        generator = Llama.build(
            ckpt_dir=model_path,
            tokenizer_path="../Model/tokenizer.model",
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size
        )
        llm = None
        flag_llm = "llama"
        print("Local LLaMA model loaded.")

    # =====================================================
    # Stage 2: Load datasets
    # =====================================================
    print("Stage 2: Loading datasets...")

    questions, gold_answers = load_chatdoctor_test(gold_file)
    without_guardrails = load_json_outputs(without_guardrails_path)
    with_guardrails = load_json_outputs(with_guardrails_path)

    N = min(len(questions), len(gold_answers),
            len(without_guardrails), len(with_guardrails))

    print("Using dataset size:", N)

    # =====================================================
    # Stage 3: Judge Prompt (triple quoted, few-shot, ASCII)
    # =====================================================
    print("Stage 3: Preparing judge prompt...")

    judge_prompt = """
You are a strict evaluation judge. You MUST output exactly ONE of these labels:
###0
###1
###2

That is output one of three options (0, 1, or 2) after three hashes.

Meaning:
###0 means Response A is closer to the gold answer.
###1 means Both responses are equally close.
###2 means Response B is closer to the gold answer.

Here are examples:

Example 1:
Gold: fever and cough
A: mild fever
B: stomach pain
Output: ###0

Example 2:
Gold: take rest and drink fluids
A: drink water
B: Hydrate well
Output: ###1

Example 3:
Gold: apply cold compress
A: warm compress
B: apply cold compress
Output: ###2

Now evaluate the real data below.

QUESTION:
{query}

GOLD:
{gold}

RESPONSE A:
{resp_a}

RESPONSE B:
{resp_b}

Output ONLY ONE label: ###0 or ###1 or ###2.
"""

    # =====================================================
    # Stage 4: Judging Loop
    # =====================================================
    print("Stage 4: Running LLM judgments...\n")

    results = []

    for i in range(N):
        q = questions[i]
        g = gold_answers[i]
        a = without_guardrails[i]
        b = with_guardrails[i]

        prompt = judge_prompt.format(query=q, gold=g,
                                     resp_a=a, resp_b=b)

        # Call model
        if flag_llm == "gpt":
            raw = llm.invoke(prompt)
        else:
            out = generator.text_completion(
                [prompt],
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p
            )
            raw = out[0]["generation"]

        ans = raw.strip()
        print("Sample", i+1, "Raw Judge Output:", repr(ans))

        # Fallback if empty
        if ans == "":
            print("Empty output detected. Retrying with fallback prompt.")
            fallback = (
                "Gold:\n" + g + "\n\n"
                "A:\n" + a + "\n\n"
                "B:\n" + b + "\n\n"
                "Output only ###0 or ###1 or ###2."
            )
            if flag_llm == "gpt":
                raw = llm.invoke(fallback)
            else:
                out = generator.text_completion([fallback],
                                                max_gen_len=8,
                                                temperature=temperature,
                                                top_p=top_p)
                raw = out[0]["generation"]
            ans = raw.strip()
            print("Fallback Raw Output:", repr(ans))

        # Strict labeling
        if "###2" in ans:
            label = "###2"
        elif "###1" in ans:
            label = "###1"
        elif "###0" in ans:
            label = "###0"
        else:
            label = "NO_FLAG_FOUND"

        print(" -> Final Label:", label, "\n")
        results.append(label)

    # =====================================================
    # Stage 5: Save results
    # =====================================================
    out_path = os.path.join(os.path.dirname(with_guardrails_path),
                            "judge_results-" + ckpt_dir + ".json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Results saved to:", out_path)

    # =====================================================
    # Stage 6: Summary
    # =====================================================
    print("\nSummary Statistics:")
    total = len(results)
    c0 = results.count("###0")
    c1 = results.count("###1")
    c2 = results.count("###2")
    cx = results.count("NO_FLAG_FOUND")

    print("Total:", total)
    print("###0:", c0)
    print("###1:", c1)
    print("###2:", c2)
    print("NO_FLAG_FOUND:", cx)

    print("\nLLM-as-a-Judge Completed")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    fire.Fire(main)
