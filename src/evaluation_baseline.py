import json
import re
import os
import argparse
from nltk.tokenize import RegexpTokenizer
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import torch
from bert_score import score as bertscore


def load_chatdoctor_test(file_path):
    """
    Loads ChatDoctor test set and returns (questions, ground_truth_answers)
    Format expected:
        input: <question>
        output: <answer>
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read().strip().split("\n\n")

    questions, answers = [], []
    for item in data:
        input_match = re.search(r"input:\s*(.*)", item, re.DOTALL)
        output_match = re.search(r"output:\s*(.*)", item, re.DOTALL)
        if input_match and output_match:
            questions.append(input_match.group(1).strip())
            answers.append(output_match.group(1).strip())
    return questions, answers

def fuzzy_match(base_prefix: str) -> str:
    """
    base_prefix: string like "Inputs&Outputs/exp1/baseline/outputs"
    returns the full path to the matched file: e.g. ".../outputs-final.json"
    """
    directory = os.path.dirname(base_prefix)  # remove the trailing "outputs"
    prefix = os.path.basename(base_prefix)    # usually "outputs"

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    # Regex: prefix + -anything.json   OR prefix.json
    pattern = re.compile(rf"^{re.escape(prefix)}(-.*)?\.json$")

    for fname in os.listdir(directory):
        if pattern.match(fname):
            return os.path.join(directory, fname)

    raise FileNotFoundError(f"No file matching pattern '{prefix}*.json' found in {directory}")

def load_generated_outputs(exp_name):
    """
    Loads generated outputs from Inputs&Outputs/<exp_name>/outputs.json
    """
    path = fuzzy_match(f"Inputs&Outputs/{exp_name}/baseline/outputs")
    with open(path, "r", encoding="utf-8") as f:
        outputs = json.load(f)
    return outputs


def compute_all_metrics(refs, hyps,
                        embed_model="sentence-transformers/all-MiniLM-L6-v2",
                        bert_model="microsoft/deberta-xlarge-mnli"):
    """
    Computes ROUGE-L, semantic cosine similarity, BERTScore F1, and exact match.
    """
    total = len(refs)
    tokenizer = RegexpTokenizer(r"\w+")
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # --- 1. ROUGE-L ---
    total_rouge = 0.0
    exact_matches = 0
    for ref, hyp in zip(refs, hyps):
        scores = rouge.score(ref, hyp)
        total_rouge += scores["rougeL"].fmeasure
        if tokenizer.tokenize(ref.lower()) == tokenizer.tokenize(hyp.lower()):
            exact_matches += 1
    avg_rouge = total_rouge / total if total else 0.0
    exact_match_rate = exact_matches / total if total else 0.0
    print("ROUGE done")
    # --- 2. Sentence-BERT cosine similarity ---
    print("Encoding with SentenceTransformer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(embed_model, device=device)
    ref_emb = embedder.encode(refs, convert_to_tensor=True, show_progress_bar=False)
    hyp_emb = embedder.encode(hyps, convert_to_tensor=True, show_progress_bar=False)
    semantic_sim = util.cos_sim(ref_emb, hyp_emb).diagonal().mean().item()
    print("sentence done")
    # --- 3. BERTScore F1 ---
    print("Computing BERTScore...")
    P, R, F1 = bertscore(hyps, refs, lang="en", model_type=bert_model, device=device)
    avg_bertscore = float(torch.mean(F1))
    print("Bert score done")
    return {
        "rougeL": avg_rouge,
        "semantic": semantic_sim,
        "bertscore": avg_bertscore,
        "exact": exact_match_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="Full baseline evaluation with ROUGE-L, SemanticSim, and BERTScore")
    parser.add_argument("--exp_name", type=str, required=True,
                        help="Experiment folder name under Inputs&Outputs/")
    parser.add_argument("--test_file", type=str,
                        default="../Data/chatdoctor-test/chatdoctor.txt",
                        help="Path to ChatDoctor test set")
    parser.add_argument("--embed_model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence transformer for semantic similarity")
    parser.add_argument("--bert_model", type=str,
                        default="distilbert-base-uncased",
                        help="Model for BERTScore (default = distilbert-base-uncased)")
    args = parser.parse_args()

    print(f"Evaluating experiment: {args.exp_name}")
    print(f"Using test file: {args.test_file}")
    print(f"Embedding model: {args.embed_model}")
    print(f"BERTScore model: {args.bert_model}")

    # Load reference and generated data
    questions, refs = load_chatdoctor_test(args.test_file)
    hyps = load_generated_outputs(args.exp_name)

    # Align lengths
    N = min(len(refs), len(hyps))
    refs, hyps = refs[:N], hyps[:N]
    print(f"Total samples: {N}")

    # Compute metrics
    metrics = compute_all_metrics(refs, hyps, args.embed_model, args.bert_model)

    # Print results
    print("\n==== Baseline Evaluation Results ====")
    print(f"Average ROUGE-L F1:         {metrics['rougeL']:.4f}")
    print(f"Average Semantic Similarity: {metrics['semantic']:.4f}")
    print(f"Average BERTScore F1:       {metrics['bertscore']:.4f}")
    print(f"Exact Match Rate:           {metrics['exact']*100:.2f}%")
    print("=====================================\n")

    # Optional: show sample outputs
    for i in range(min(3, N)):
        print(f"\n--- Example {i+1} ---")
        print(f"Q: {questions[i]}")
        print(f"Reference: {refs[i]}")
        print(f"Generated: {hyps[i]}")


if __name__ == "__main__":
    main()

