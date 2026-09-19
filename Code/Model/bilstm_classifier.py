import os
import json
import argparse
import random
import time
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.feature_extraction.text import CountVectorizer


TARGET_CLASSES = ["benign", "direct_injection", "indirect_injection"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(dataset_dir):
    clean_dir = os.path.join(dataset_dir, "clean_data_dir")
    files = {
        "train": os.path.join(clean_dir, "train-00000-of-00001.parquet"),
        "validation": os.path.join(clean_dir, "validation-00000-of-00001.parquet"),
        "test": os.path.join(clean_dir, "test-00000-of-00001.parquet")
    }
    data = {}
    for split, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {split} file: {path}")
        data[split] = pd.read_parquet(path)
    return data


def prepare_data(data, text_column, category_column):
    label2id = {label: i for i, label in enumerate(TARGET_CLASSES)}
    id2label = {i: label for label, i in label2id.items()}

    for split, df in data.items():
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in {split} dataset.")
        if category_column not in df.columns:
            raise ValueError(f"Column '{category_column}' not found in {split} dataset.")

        df = df.copy()
        df[category_column] = df[category_column].astype(str).str.strip().str.lower()
        df = df[df[category_column].isin(TARGET_CLASSES)].copy()
        df[text_column] = df[text_column].fillna("").astype(str)
        df = df[df[text_column].str.strip().str.len() > 0].copy()
        df["labels"] = df[category_column].map(label2id).astype(int)
        data[split] = df.reset_index(drop=True)

    return data, label2id, id2label


def build_vocabulary(train_texts, max_vocab_size=30000, min_freq=2):
    vectorizer = CountVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        max_features=max_vocab_size,
        min_df=min_freq
    )
    vectorizer.fit(train_texts)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1]):
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def encode_text(text, vocab, max_length):
    tokens = text.lower().split()
    ids = [vocab.get(token, 1) for token in tokens[:max_length]]
    if len(ids) < max_length:
        ids.extend([0] * (max_length - len(ids)))
    return ids


class PromptDataset(Dataset):
    def __init__(self, df, text_column, vocab, max_length):
        self.texts = df[text_column].tolist()
        self.labels = df["labels"].tolist()
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(
                encode_text(self.texts[idx], self.vocab, self.max_length),
                dtype=torch.long
            ),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        hidden_dim=128,
        num_layers=1,
        dropout=0.3,
        num_classes=3
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)

        mask = input_ids.ne(0).unsqueeze(-1)
        lengths = mask.sum(dim=1).clamp(min=1)
        x = (x * mask).sum(dim=1) / lengths

        return self.classifier(self.dropout(x))


def calculate_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = len(labels) / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def format_time(seconds):
    """Formats elapsed/remaining seconds into mm:ss or hh:mm:ss."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    else:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def print_progress_bar(iteration, total, prefix='', suffix='', length=30, fill='█', elapsed=0):
    """Custom clean progress bar with ETA support."""
    if total <= 0:
        total = 1
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    
    # Calculate ETA
    eta_str = ""
    if iteration > 0 and elapsed > 0:
        time_per_iter = elapsed / iteration
        remaining_iters = total - iteration
        eta = remaining_iters * time_per_iter
        eta_str = f" | ETA: {format_time(eta)}"
    
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% ({iteration}/{total}) {suffix}{eta_str}')
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    total_batches = len(loader)
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            total_loss += loss.item() * len(labels)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            elapsed = time.time() - start_time
            print_progress_bar(batch_idx + 1, total_batches, prefix="Evaluating", length=25, elapsed=elapsed)

    return (
        total_loss / len(loader.dataset),
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )


def metrics_from_predictions(labels, preds):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "weighted_f1": weighted_f1
    }


def plot_training_history(history, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    ax.plot(epochs, history["val_loss"], marker="o", label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("BiLSTM Training History")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_history.png"), dpi=300)
    plt.close(fig)


def plot_confusion_matrix(labels, preds, id2label, output_dir):
    cm = confusion_matrix(labels, preds, labels=list(id2label.keys()))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(id2label)))
    ax.set_yticks(range(len(id2label)))
    ax.set_xticklabels([id2label[i] for i in id2label], rotation=30, ha="right")
    ax.set_yticklabels([id2label[i] for i in id2label])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("BiLSTM Confusion Matrix")

    for i in range(len(id2label)):
        for j in range(len(id2label)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close(fig)


def plot_roc(labels, probs, id2label, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    valid_curves = 0

    for class_id, label in id2label.items():
        y_true = (labels == class_id).astype(int)
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, probs[:, class_id])
        auc = roc_auc_score(y_true, probs[:, class_id])
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
        valid_curves += 1

    if valid_curves:
        ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("BiLSTM One-vs-Rest ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=300)
    plt.close(fig)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    total_batches = len(loader)
    start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)

        elapsed = time.time() - start_time
        print_progress_bar(batch_idx + 1, total_batches, prefix="Training", length=25, elapsed=elapsed)

    return total_loss / len(loader.dataset)


def parse_args():
    parser = argparse.ArgumentParser(description="BiLSTM Prompt Injection Classifier")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--text_column", required=True)
    parser.add_argument("--category_column", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_vocab_size", type=int, default=30000)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_class_weights", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir or os.path.join(args.dataset_dir, "bilstm_result")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("BiLSTM Prompt Classifier")
    print("=" * 70)
    print(f"Dataset:       {args.dataset_dir}")
    print("Model:         BiLSTM")
    print(f"Text column:   {args.text_column}")
    print(f"Category:      {args.category_column}")
    print(f"Max length:    {args.max_length}")
    print(f"Vocabulary:    {args.max_vocab_size}")
    print(f"Embedding:     {args.embedding_dim}")
    print(f"Hidden size:   {args.hidden_dim}")
    print(f"Epochs:        {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Class weights: {not args.no_class_weights}")
    print(f"Device:        {device}")

    data = load_data(args.dataset_dir)
    data, label2id, id2label = prepare_data(
        data, args.text_column, args.category_column
    )

    print("\nDataset sizes:")
    for split, df in data.items():
        print(f"  {split}: {len(df):,}")

    print("\nClass distribution:")
    for split, df in data.items():
        counts = df["labels"].value_counts().sort_index()
        print(
            f"  {split}: " +
            ", ".join(
                f"{id2label[i]}={counts.get(i, 0):,}"
                for i in id2label
            )
        )

    vocab = build_vocabulary(
        data["train"][args.text_column],
        args.max_vocab_size,
        args.min_freq
    )
    print(f"\nVocabulary size: {len(vocab):,}")

    datasets = {
        split: PromptDataset(
            df, args.text_column, vocab, args.max_length
        )
        for split, df in data.items()
    }

    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=torch.cuda.is_available()
        ),
        "validation": DataLoader(
            datasets["validation"], batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available()
        ),
        "test": DataLoader(
            datasets["test"], batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available()
        )
    }

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=len(TARGET_CLASSES)
    ).to(device)

    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
        class_weights = None
    else:
        class_weights = calculate_class_weights(
            data["train"]["labels"].values, len(TARGET_CLASSES)
        ).to(device)
        print(f"\nClass weights: {class_weights.cpu().tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": []
    }

    best_f1 = -1.0
    best_path = os.path.join(output_dir, "best_model.pt")

    print("\nTraining...")
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        train_loss = train_epoch(
            model, loaders["train"], optimizer, criterion, device
        )
        val_loss, val_labels, val_preds, _ = evaluate(
            model, loaders["validation"], criterion, device
        )
        val_metrics = metrics_from_predictions(val_labels, val_preds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])

        print(
            f"Epoch {epoch + 1}/{args.epochs} Summary -> "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
                "label2id": label2id,
                "id2label": id2label,
                "config": vars(args)
            }, best_path)

    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    plot_training_history(history, output_dir)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\n--- Test Set Evaluation ---")
    test_loss, labels, preds, probs = evaluate(
        model, loaders["test"], criterion, device
    )
    test_metrics = metrics_from_predictions(labels, preds)

    print("\n" + "=" * 70)
    print("Test Results")
    print("=" * 70)
    print(f"Test loss:      {test_loss:.4f}")
    print(f"Accuracy:       {test_metrics['accuracy']:.4f}")
    print(f"Macro precision:{test_metrics['macro_precision']:.4f}")
    print(f"Macro recall:   {test_metrics['macro_recall']:.4f}")
    print(f"Macro F1:       {test_metrics['macro_f1']:.4f}")
    print(f"Weighted F1:    {test_metrics['weighted_f1']:.4f}")

    report = classification_report(
        labels,
        preds,
        labels=list(id2label.keys()),
        target_names=list(id2label.values()),
        digits=4,
        zero_division=0
    )
    print("\nClassification Report:\n")
    print(report)

    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    try:
        macro_auc = roc_auc_score(
            labels, probs, multi_class="ovr", average="macro"
        )
        weighted_auc = roc_auc_score(
            labels, probs, multi_class="ovr", average="weighted"
        )
    except ValueError:
        macro_auc = None
        weighted_auc = None

    print(f"Macro ROC-AUC:  {macro_auc:.4f}" if macro_auc is not None else "Macro ROC-AUC: N/A")
    print(f"Weighted ROC-AUC: {weighted_auc:.4f}" if weighted_auc is not None else "Weighted ROC-AUC: N/A")

    results = {
        **test_metrics,
        "test_loss": test_loss,
        "macro_roc_auc": macro_auc,
        "weighted_roc_auc": weighted_auc,
        "best_validation_macro_f1": best_f1,
        "n_test_samples": len(labels),
        "classes": id2label,
        "config": vars(args)
    }

    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    predictions = data["test"][[args.text_column, args.category_column]].copy()
    predictions["true_label"] = [id2label[int(x)] for x in labels]
    predictions["predicted_label"] = [id2label[int(x)] for x in preds]

    for class_id, label in id2label.items():
        predictions[f"prob_{label}"] = probs[:, class_id]

    predictions.to_parquet(
        os.path.join(output_dir, "test_predictions.parquet"),
        index=False
    )

    plot_confusion_matrix(labels, preds, id2label, output_dir)
    plot_roc(labels, probs, id2label, output_dir)

    print(f"\nResults saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()