import re
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "clean_data_dir/full_dataset.parquet"
OUTPUT_DIR = Path("clean_data_dir/indirect_detection")

CANDIDATE_FILE = OUTPUT_DIR / "indirect_candidates.parquet"
REVIEW_FILE = OUTPUT_DIR / "indirect_review_results.parquet"
FINAL_FILE = OUTPUT_DIR / "three_class_full.parquet"
AUDIT_FILE = OUTPUT_DIR / "indirect_detection_audit.csv"

INDIRECT_CATEGORIES = {"indirect_injection"}

SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MIN_SEED_COUNT = 3
SEED_PHRASE_MIN_DF = 2
SEED_PHRASE_MAX_DF = 0.80
TOP_SEED_PHRASES = 100

SEMANTIC_THRESHOLD = 0.52
STRONG_SEMANTIC_THRESHOLD = 0.68

CANDIDATE_SCORE_THRESHOLD = 5.0

AUTO_INDIRECT_THRESHOLD = 9.0
AUTO_BENIGN_THRESHOLD = -3.0

REVIEW_EVERY = 10


# ============================================================
# HANDCRAFTED SIGNALS
# ============================================================

CARRIER_PATTERNS = {
    "email": r"\b(?:email|e-mail|message|mail|inbox|subject line|sender)\b",
    "document": r"\b(?:document|file|pdf|report|attachment|spreadsheet|memo)\b",
    "webpage": r"\b(?:webpage|website|web page|html|page content|online article)\b",
    "content": r"\b(?:content|text|paragraph|article|post|comment|review|description)\b",
    "data": r"\b(?:data|dataset|record|database|entry|metadata|field)\b",
    "retrieval": r"\b(?:retrieved|retrieval|search result|search results|knowledge base|context)\b",
    "tool": r"\b(?:tool output|tool result|api response|api output|function result)\b",
    "external": r"\b(?:external source|third[- ]party|uploaded|user[- ]provided)\b"
}

MODEL_DIRECTIVE_PATTERNS = {
    "ai_reference": r"\b(?:ai|assistant|chatbot|chatgpt|gpt|llm|language model|agent|model)\b",
    "instruction": r"\b(?:instruction|instructions|prompt|directive|command|rule)\b",
    "ignore": r"\b(?:ignore|disregard|forget|override|bypass|do not follow)\b",
    "role": r"\b(?:you are now|act as|pretend to be|roleplay|new role)\b",
    "priority": r"\b(?:highest priority|system message|system instruction|developer instruction)\b",
    "hidden": r"\b(?:hidden instruction|secret instruction|invisible instruction|embedded instruction)\b"
}

ACTION_PATTERNS = {
    "exfiltration": r"\b(?:send|transmit|forward|upload|leak|exfiltrate|share)\b.{0,100}\b(?:data|information|secret|password|token|credential|content)\b",
    "external_request": r"\b(?:visit|open|navigate|request|fetch|call|connect)\b.{0,100}\b(?:url|website|server|endpoint|api|link)\b",
    "credential": r"\b(?:password|credential|api key|secret|token|authentication|access key)\b",
    "destructive": r"\b(?:delete|destroy|remove|wipe|erase|disable)\b",
    "execution": r"\b(?:execute|run|shell|command|code|script)\b",
    "redirect": r"\b(?:redirect|forward|send to|route to)\b"
}

DIRECT_PATTERNS = {
    "direct_ignore": r"^\s*(?:ignore|disregard|forget|override)\b",
    "direct_role": r"^\s*(?:you are now|act as|pretend to be)\b",
    "direct_command": r"^\s*(?:do|execute|perform|follow|output|write|generate)\b",
    "direct_system": r"\b(?:ignore previous instructions|ignore all previous|disregard previous)\b",
    "direct_prompt": r"^\s*(?:system|developer|assistant|user)\s*:"
}

STRONG_INDIRECT_PATTERNS = {
    "embedded_instruction": r"\b(?:embedded|hidden|concealed|inserted|injected)\b.{0,100}\b(?:instruction|prompt|directive|command)\b",
    "content_instruction": r"\b(?:document|email|webpage|article|file|content|text|page)\b.{0,120}\b(?:instructs?|tells?|commands?|directs?|asks?)\b",
    "model_instruction": r"\b(?:document|email|webpage|article|file|content|text|page)\b.{0,150}\b(?:assistant|model|agent|ai|chatbot)\b",
    "external_instruction": r"\b(?:external|third[- ]party|retrieved|uploaded|user[- ]provided)\b.{0,120}\b(?:instruction|prompt|directive|command)\b",
    "hidden_behavior": r"\b(?:hidden|embedded|concealed|invisible)\b.{0,150}\b(?:assistant|model|agent|ai)\b"
}


# ============================================================
# COMPILE PATTERNS
# ============================================================

compile_p = lambda p: {k: re.compile(v, re.I | re.S) for k, v in p.items()}

CARRIERS = compile_p(CARRIER_PATTERNS)
DIRECTIVES = compile_p(MODEL_DIRECTIVE_PATTERNS)
ACTIONS = compile_p(ACTION_PATTERNS)
DIRECTS = compile_p(DIRECT_PATTERNS)
STRONG_INDIRECT = compile_p(STRONG_INDIRECT_PATTERNS)


def find_matches(text, patterns):
    return {k: len(p.findall(text)) for k, p in patterns.items()}


def count_direct_signals(text):
    return sum(bool(p.search(text)) for p in DIRECTS.values())


# ============================================================
# SEED PATTERN LEARNING
# ============================================================

def learn_seed_phrases(seed_texts, background_texts):
    if len(seed_texts) < MIN_SEED_COUNT:
        return {}

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=SEED_PHRASE_MIN_DF,
        max_df=SEED_PHRASE_MAX_DF,
        sublinear_tf=True,
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b"
    )

    try:
        seed_matrix = vectorizer.fit_transform(seed_texts)
    except ValueError:
        return {}

    vocab = vectorizer.get_feature_names_out()
    seed_scores = np.asarray(seed_matrix.mean(axis=0)).ravel()

    if background_texts:
        bg_matrix = vectorizer.transform(background_texts)
        bg_scores = np.asarray(bg_matrix.mean(axis=0)).ravel()
    else:
        bg_scores = np.zeros_like(seed_scores)

    ratios = (seed_scores + 1e-6) / (bg_scores + 1e-5)
    scores = seed_scores * np.log1p(ratios)

    ranked = np.argsort(scores)[::-1]
    phrases = {}

    for idx in ranked[:TOP_SEED_PHRASES]:
        phrase = vocab[idx]
        score = float(scores[idx])
        if score > 0:
            phrases[phrase] = score

    return phrases


def seed_phrase_score(text, phrases):
    if not phrases:
        return 0.0, []

    low = text.lower()
    matched = []
    score = 0.0

    for phrase, weight in phrases.items():
        if re.search(r"\b" + re.escape(phrase) + r"\b", low):
            matched.append(phrase)
            score += min(weight * 2.5, 3.0)

    return min(score, 6.0), matched


# ============================================================
# SEMANTIC SEED MODEL
# ============================================================

class SeedSemanticMatcher:
    def __init__(self, seed_texts):
        self.seed_texts = seed_texts
        self.model = None
        self.seed_embeddings = None

        if len(seed_texts) >= MIN_SEED_COUNT:
            print(f"Loading semantic model: {SEMANTIC_MODEL}")
            self.model = SentenceTransformer(SEMANTIC_MODEL)
            self.seed_embeddings = self.model.encode(
                seed_texts,
                normalize_embeddings=True,
                show_progress_bar=True
            )

    def score(self, texts):
        if self.model is None or self.seed_embeddings is None:
            return np.zeros(len(texts))

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        similarities = embeddings @ self.seed_embeddings.T
        return similarities.max(axis=1)


# ============================================================
# INDIRECT SCORING
# ============================================================

def score_indirect_injection(text, seed_phrases=None, semantic_score=0.0, category=None):
    text = str(text)

    carrier = find_matches(text, CARRIERS)
    directive = find_matches(text, DIRECTIVES)
    action = find_matches(text, ACTIONS)
    strong = find_matches(text, STRONG_INDIRECT)

    carrier_n = sum(v > 0 for v in carrier.values())
    directive_n = sum(v > 0 for v in directive.values())
    action_n = sum(v > 0 for v in action.values())
    strong_n = sum(v > 0 for v in strong.values())

    direct_n = count_direct_signals(text)

    score = 0.0

    score += min(carrier_n * 2.0, 6.0)
    score += min(directive_n * 2.0, 6.0)
    score += min(action_n * 1.5, 4.5)
    score += min(strong_n * 3.0, 9.0)

    combo = carrier_n > 0 and directive_n > 0
    if combo:
        score += 2.0

    if carrier_n > 0 and directive_n > 0 and action_n > 0:
        score += 3.0

    phrase_score, matched_phrases = seed_phrase_score(text, seed_phrases or {})
    score += phrase_score

    if semantic_score >= STRONG_SEMANTIC_THRESHOLD:
        score += 5.0
    elif semantic_score >= SEMANTIC_THRESHOLD:
        score += 2.5

    if category in INDIRECT_CATEGORIES:
        score += 3.0
    elif category in {"adversarial", "prompt_injection", "payload_injection", "output_manipulation", "response_manipulation", "system_manipulation"}:
        score += 1.0

    score -= min(direct_n * 2.5, 7.5)

    if direct_n >= 2 and carrier_n == 0:
        score -= 3.0

    return {
        "score": round(score, 4),
        "carrier_count": carrier_n,
        "directive_count": directive_n,
        "action_count": action_n,
        "strong_indirect_count": strong_n,
        "direct_count": direct_n,
        "semantic_similarity": round(float(semantic_score), 4),
        "seed_phrase_score": round(float(phrase_score), 4),
        "matched_seed_phrases": matched_phrases,
        "carrier_matches": [k for k, v in carrier.items() if v],
        "directive_matches": [k for k, v in directive.items() if v],
        "action_matches": [k for k, v in action.items() if v],
        "strong_indirect_matches": [k for k, v in strong.items() if v]
    }


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print(f"Loading: {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)

    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["category"] = df["category"].fillna("").astype(str)

    print(f"Loaded {len(df):,} rows")
    return df


# ============================================================
# BUILD SEED POOL
# ============================================================

def build_seed_pool(df):
    seed_mask = df["category"].str.lower().isin(INDIRECT_CATEGORIES)
    seeds = df.loc[seed_mask, "text"].drop_duplicates().tolist()

    print(f"Known indirect-category seeds: {len(seeds)}")

    if len(seeds) < MIN_SEED_COUNT:
        print("WARNING: Too few indirect seeds for reliable seed learning.")

    confirmed_file = REVIEW_FILE

    if confirmed_file.exists():
        reviewed = pd.read_parquet(confirmed_file)
        if "review_label" in reviewed.columns:
            extra = reviewed.loc[reviewed["review_label"] == 2, "text"].drop_duplicates().tolist()
            if extra:
                seeds = list(dict.fromkeys(seeds + extra))
                print(f"Added manually confirmed indirect seeds: {len(extra)}")

    return seeds


# ============================================================
# GENERATE CANDIDATES
# ============================================================

def generate_candidates(df):
    seed_texts = build_seed_pool(df)

    background_mask = ~df.index.isin(
        df.index[df["category"].str.lower().isin(INDIRECT_CATEGORIES)]
    )

    background_texts = df.loc[background_mask, "text"].tolist()

    print("Learning seed-specific phrases...")
    seed_phrases = learn_seed_phrases(seed_texts, background_texts)

    print(f"Learned seed phrases: {len(seed_phrases)}")

    if seed_phrases:
        print("Top learned phrases:")
        for phrase, score in list(seed_phrases.items())[:25]:
            print(f"  {phrase}: {score:.4f}")

    matcher = SeedSemanticMatcher(seed_texts)

    print("Calculating semantic similarity...")
    semantic_scores = matcher.score(df["text"].tolist())

    records = []

    for i, row in df.iterrows():
        category = str(row.get("category", ""))
        semantic = float(semantic_scores[df.index.get_loc(i)])

        result = score_indirect_injection(
            row["text"],
            seed_phrases,
            semantic,
            category.lower()
        )

        is_known_seed = category.lower() in INDIRECT_CATEGORIES

        candidate = (
            result["score"] >= CANDIDATE_SCORE_THRESHOLD or
            result["strong_indirect_count"] > 0 or
            result["semantic_similarity"] >= STRONG_SEMANTIC_THRESHOLD or
            is_known_seed
        )

        if not candidate:
            continue

        records.append({
            "original_index": i,
            "text": row["text"],
            "label": row.get("label", None),
            "category": category,
            "source": row.get("source", None),
            "severity": row.get("severity", None),
            "group_id": row.get("group_id", None),
            "augmented": row.get("augmented", None),
            "token_count": row.get("token_count", None),
            "indirect_score": result["score"],
            "semantic_similarity": result["semantic_similarity"],
            "seed_phrase_score": result["seed_phrase_score"],
            "carrier_count": result["carrier_count"],
            "directive_count": result["directive_count"],
            "action_count": result["action_count"],
            "strong_indirect_count": result["strong_indirect_count"],
            "direct_count": result["direct_count"],
            "matched_seed_phrases": json.dumps(result["matched_seed_phrases"]),
            "carrier_matches": json.dumps(result["carrier_matches"]),
            "directive_matches": json.dumps(result["directive_matches"]),
            "action_matches": json.dumps(result["action_matches"]),
            "strong_indirect_matches": json.dumps(result["strong_indirect_matches"]),
            "known_indirect_seed": is_known_seed
        })

    candidates = pd.DataFrame(records)

    if candidates.empty:
        return candidates

    candidates = candidates.sort_values(
        ["known_indirect_seed", "indirect_score", "semantic_similarity"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return candidates


# ============================================================
# REVIEW
# ============================================================

def print_candidate(row, number, total):
    print("\n" + "=" * 90)
    print(f"[{number}/{total}]")
    print(f"Score: {row['indirect_score']:.2f}")
    print(f"Semantic similarity: {row['semantic_similarity']:.3f}")
    print(f"Seed phrase score: {row['seed_phrase_score']:.2f}")
    print(f"Category: {row['category']}")
    print(f"Known indirect seed: {row['known_indirect_seed']}")
    print(f"Carrier signals: {row['carrier_matches']}")
    print(f"Directive signals: {row['directive_matches']}")
    print(f"Action signals: {row['action_matches']}")
    print(f"Strong indirect signals: {row['strong_indirect_matches']}")
    print(f"Matched seed phrases: {row['matched_seed_phrases']}")
    print("-" * 90)
    print(row["text"])
    print("=" * 90)


def review_candidates(candidates):
    if candidates.empty:
        print("No candidates found.")
        return candidates

    candidates = candidates.copy()

    if REVIEW_FILE.exists():
        old = pd.read_parquet(REVIEW_FILE)

        if "review_label" in old.columns and "original_index" in old.columns:
            existing = dict(zip(old["original_index"], old["review_label"]))
            candidates["review_label"] = candidates["original_index"].map(existing)
        else:
            candidates["review_label"] = np.nan
    else:
        candidates["review_label"] = np.nan

    total = len(candidates)

    print("\nReview labels:")
    print("  i = indirect")
    print("  d = direct")
    print("  b = benign")
    print("  s = skip")
    print("  q = quit and save")
    print()

    reviewed_since_save = 0

    for idx in range(total):
        if pd.notna(candidates.loc[idx, "review_label"]):
            continue

        row = candidates.loc[idx]
        print_candidate(row, idx + 1, total)

        if row["known_indirect_seed"]:
            print("This is already in an indirect-category seed set.")

        if row["indirect_score"] >= AUTO_INDIRECT_THRESHOLD:
            print("Suggested: INDIRECT")
        elif row["indirect_score"] <= AUTO_BENIGN_THRESHOLD:
            print("Suggested: BENIGN")
        else:
            print("Suggested: REVIEW")

        while True:
            choice = input("\nLabel [i/d/b/s/q]: ").strip().lower()

            if choice == "i":
                candidates.loc[idx, "review_label"] = 2
                break
            if choice == "d":
                candidates.loc[idx, "review_label"] = 1
                break
            if choice == "b":
                candidates.loc[idx, "review_label"] = 0
                break
            if choice == "s":
                break
            if choice == "q":
                candidates.to_parquet(REVIEW_FILE, index=False)
                print(f"Saved review progress to {REVIEW_FILE}")
                return candidates

        reviewed_since_save += 1

        if reviewed_since_save >= REVIEW_EVERY:
            candidates.to_parquet(REVIEW_FILE, index=False)
            reviewed_since_save = 0
            print(f"Progress saved: {REVIEW_FILE}")

    candidates.to_parquet(REVIEW_FILE, index=False)
    print(f"\nReview complete: {REVIEW_FILE}")

    return candidates


# ============================================================
# BUILD THREE-CLASS DATASET
# ============================================================

def build_three_class_dataset(df, reviewed):
    result = df.copy()
    result["three_class_label"] = result["label"].astype(int)

    reviewed = reviewed.dropna(subset=["review_label"]).copy()

    review_map = dict(zip(reviewed["original_index"], reviewed["review_label"]))

    for original_index, review_label in review_map.items():
        if original_index in result.index:
            result.loc[original_index, "three_class_label"] = int(review_label)

    result["three_class_label"] = result["three_class_label"].astype(int)

    result.to_parquet(FINAL_FILE, index=False)

    return result


# ============================================================
# AUDIT
# ============================================================

def create_audit(df, candidates, reviewed, final_df):
    audit = {
        "input_rows": len(df),
        "known_indirect_category_rows": int(df["category"].str.lower().isin(INDIRECT_CATEGORIES).sum()),
        "candidate_rows": len(candidates),
        "reviewed_rows": int(reviewed["review_label"].notna().sum()) if "review_label" in reviewed else 0,
        "confirmed_indirect": int((reviewed["review_label"] == 2).sum()) if "review_label" in reviewed else 0,
        "confirmed_direct": int((reviewed["review_label"] == 1).sum()) if "review_label" in reviewed else 0,
        "confirmed_benign": int((reviewed["review_label"] == 0).sum()) if "review_label" in reviewed else 0,
        "final_benign": int((final_df["three_class_label"] == 0).sum()),
        "final_direct": int((final_df["three_class_label"] == 1).sum()),
        "final_indirect": int((final_df["three_class_label"] == 2).sum()),
        "semantic_threshold": SEMANTIC_THRESHOLD,
        "strong_semantic_threshold": STRONG_SEMANTIC_THRESHOLD,
        "candidate_score_threshold": CANDIDATE_SCORE_THRESHOLD,
        "auto_indirect_threshold": AUTO_INDIRECT_THRESHOLD,
        "auto_benign_threshold": AUTO_BENIGN_THRESHOLD
    }

    pd.DataFrame([audit]).to_csv(AUDIT_FILE, index=False)

    print("\nAudit:")
    for k, v in audit.items():
        print(f"  {k}: {v}")


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    print("\n" + "=" * 70)
    print("SEED-BASED INDIRECT INJECTION DETECTION")
    print("=" * 70)

    candidates = generate_candidates(df)

    if candidates.empty:
        print("No candidates generated.")
        return

    candidates.to_parquet(CANDIDATE_FILE, index=False)

    print(f"\nCandidates: {len(candidates):,}")
    print(f"Saved: {CANDIDATE_FILE}")

    print("\nCandidate distribution:")
    print(candidates["category"].value_counts().head(20))

    reviewed = review_candidates(candidates)

    reviewed.to_parquet(REVIEW_FILE, index=False)

    final_df = build_three_class_dataset(df, reviewed)

    print(f"\nThree-class dataset saved: {FINAL_FILE}")

    create_audit(df, candidates, reviewed, final_df)

    print("\nFinal labels:")
    print(final_df["three_class_label"].value_counts().sort_index())
    print("\n0 = benign")
    print("1 = direct injection")
    print("2 = confirmed indirect injection")


if __name__ == "__main__":
    main()