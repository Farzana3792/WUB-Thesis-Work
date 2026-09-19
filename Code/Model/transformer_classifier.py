import os
import json
import argparse
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize


TARGET_CLASSES = ["benign", "direct_injection", "indirect_injection"]
DEFAULT_MODEL = 'answerdotai/ModernBERT-base'
TRAIN_FILE = 'train-00000-of-00001.parquet'
VALID_FILE = 'validation-00000-of-00001.parquet'
TEST_FILE = 'test-00000-of-00001.parquet'


def load_data(dataset_dir):
    data_dir = os.path.join(dataset_dir, 'clean_data_dir')
    paths = {
        'train': os.path.join(data_dir, TRAIN_FILE),
        'validation': os.path.join(data_dir, VALID_FILE),
        'test': os.path.join(data_dir, TEST_FILE)
    }

    for split, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f'{split} file not found: {path}')

    return {
        split: pd.read_parquet(path)
        for split, path in paths.items()
    }


def prepare_labels(data, category_column):
    for split, df in data.items():
        if category_column not in df.columns:
            raise ValueError(f"Column '{category_column}' not found in {split} dataset.")

        df = df.copy()
        df[category_column] = df[category_column].astype(str).str.strip().str.lower()
        df = df[df[category_column].isin(TARGET_CLASSES)].copy()
        df["labels"] = df[category_column].map({label: i for i, label in enumerate(TARGET_CLASSES)})
        data[split] = df

    label2id = {label: i for i, label in enumerate(TARGET_CLASSES)}
    id2label = {i: label for label, i in label2id.items()}

    print(f"\nClasses ({len(TARGET_CLASSES)}):")
    for i, label in id2label.items():
        print(f"  {i}: {label}")

    return data, label2id, id2label




def prepare_text(data, text_column):
    for split, df in data.items():
        if text_column not in df.columns:
            raise ValueError(
                f"Column '{text_column}' not found in {split} dataset. "
                f'Available columns: {list(df.columns)}'
            )

        df[text_column] = df[text_column].fillna('').astype(str)
        df = df[df[text_column].str.strip().str.len() > 0].copy()
        data[split] = df.reset_index(drop=True)

    return data


def print_distribution(data, id2label):
    print('\nClass distribution:')
    for split, df in data.items():
        counts = df['labels'].value_counts().sort_index()
        print(f'\n{split.capitalize()} ({len(df):,} rows)')
        for label_id, count in counts.items():
            label = id2label[int(label_id)]
            pct = count / len(df) * 100
            print(f'  {label}: {count:,} ({pct:.2f}%)')


def build_datasets(data, text_column):
    datasets = {}

    for split, df in data.items():
        columns = [text_column, 'labels']
        datasets[split] = Dataset.from_pandas(
            df[columns], preserve_index=False
        )

    return datasets


def tokenize_datasets(datasets, tokenizer, text_column, max_length):
    def tokenize(batch):
        return tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_length
        )

    tokenized = {}
    for split, dataset in datasets.items():
        tokenized[split] = dataset.map(
            tokenize,
            batched=True,
            remove_columns=[text_column]
        )

    return tokenized


def calculate_class_weights(train_dataset, num_labels):
    labels = np.array(train_dataset['labels'])
    counts = np.bincount(labels, minlength=num_labels)
    weights = len(labels) / (num_labels * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float)


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


class MetricsLogger(TrainerCallback):
    def __init__(self):
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            row = {'step': state.global_step}
            row.update(logs)
            self.history.append(row)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = torch.softmax(
        torch.tensor(logits), dim=1
    ).numpy()
    predictions = probabilities.argmax(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )

    weighted_f1 = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )[2]

    return {
        'accuracy': accuracy_score(labels, predictions),
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': f1,
        'weighted_f1': weighted_f1
    }


def save_training_history(history, result_dir):
    if not history:
        return

    history = pd.DataFrame(history)
    history.to_csv(
        os.path.join(result_dir, 'training_history.csv'),
        index=False
    )

    if 'loss' in history.columns:
        train_loss = history.dropna(subset=['loss'])

        if len(train_loss):
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(
                train_loss['step'],
                train_loss['loss'],
                marker='o',
                label='Training loss'
            )

            eval_loss = history.dropna(subset=['eval_loss'])
            if len(eval_loss):
                ax.plot(
                    eval_loss['step'],
                    eval_loss['eval_loss'],
                    marker='o',
                    label='Validation loss'
                )

            ax.set_title('Training and Validation Loss')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=.3)
            plt.tight_layout()
            fig.savefig(
                os.path.join(result_dir, 'training_loss.png'),
                dpi=300,
                bbox_inches='tight'
            )
            plt.close(fig)

    eval_rows = history.dropna(
        subset=['eval_macro_f1'], how='any'
    )

    if len(eval_rows):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(
            eval_rows['step'],
            eval_rows['eval_macro_f1'],
            marker='o'
        )
        ax.set_title('Validation Macro-F1')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Macro-F1')
        ax.grid(True, alpha=.3)
        plt.tight_layout()
        fig.savefig(
            os.path.join(result_dir, 'validation_macro_f1.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close(fig)


def evaluate_test(
    trainer, test_dataset, id2label, test_df, result_dir
):
    output = trainer.predict(test_dataset)
    logits = output.predictions
    labels = output.label_ids

    probabilities = torch.softmax(
        torch.tensor(logits), dim=1
    ).numpy()
    predictions = probabilities.argmax(axis=1)

    label_ids = sorted(id2label.keys())
    target_names = [id2label[i] for i in label_ids]

    report = classification_report(
        labels,
        predictions,
        labels=label_ids,
        target_names=target_names,
        digits=4,
        zero_division=0
    )

    with open(
        os.path.join(result_dir, 'classification_report.txt'),
        'w',
        encoding='utf-8'
    ) as f:
        f.write(report)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=label_ids,
        zero_division=0
    )

    metrics = {
        'accuracy': float(accuracy_score(labels, predictions)),
        'macro_precision': float(np.mean(precision)),
        'macro_recall': float(np.mean(recall)),
        'macro_f1': float(np.mean(f1)),
        'weighted_f1': float(
            precision_recall_fscore_support(
                labels, predictions,
                average='weighted',
                zero_division=0
            )[2]
        )
    }

    y_binary = label_binarize(labels, classes=label_ids)

    try:
        metrics['macro_auc_ovr'] = float(
            roc_auc_score(
                y_binary,
                probabilities,
                multi_class='ovr',
                average='macro'
            )
        )
        metrics['weighted_auc_ovr'] = float(
            roc_auc_score(
                y_binary,
                probabilities,
                multi_class='ovr',
                average='weighted'
            )
        )
    except ValueError:
        metrics['macro_auc_ovr'] = None
        metrics['weighted_auc_ovr'] = None

    metrics['per_class'] = {
        id2label[i]: {
            'precision': float(precision[j]),
            'recall': float(recall[j]),
            'f1': float(f1[j]),
            'support': int(support[j])
        }
        for j, i in enumerate(label_ids)
    }

    with open(
        os.path.join(result_dir, 'test_metrics.json'),
        'w',
        encoding='utf-8'
    ) as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(
        labels, predictions, labels=label_ids
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax
    )
    ax.set_title('Test Set Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.tight_layout()
    fig.savefig(
        os.path.join(result_dir, 'confusion_matrix.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))

    for j, label_id in enumerate(label_ids):
        fpr, tpr, _ = roc_curve(
            y_binary[:, j], probabilities[:, j]
        )
        class_auc = auc(fpr, tpr)
        ax.plot(
            fpr, tpr,
            linewidth=2,
            label=f'{id2label[label_id]} (AUC = {class_auc:.4f})'
        )

    ax.plot(
        [0, 1], [0, 1],
        linestyle='--',
        linewidth=1.5,
        label='Random classifier'
    )

    ax.set_title('Test Set ROC Curves')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=.3)
    plt.tight_layout()
    fig.savefig(
        os.path.join(result_dir, 'roc_auc_curves.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    predictions_df = test_df.copy()
    predictions_df['true_label_id'] = labels
    predictions_df['predicted_label_id'] = predictions

    predictions_df['true_label'] = [
        id2label[int(x)] for x in labels
    ]
    predictions_df['predicted_label'] = [
        id2label[int(x)] for x in predictions
    ]

    for i, label_id in enumerate(label_ids):
        predictions_df[
            f'prob_{id2label[label_id]}'
        ] = probabilities[:, i]

    predictions_df.to_parquet(
        os.path.join(result_dir, 'test_predictions.parquet'),
        index=False
    )

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fine-tune a Transformer for 3-class prompt classification.'
    )

    parser.add_argument(
        '--dataset_dir',
        required=True,
        help='Dataset directory, e.g. dataset_2'
    )

    parser.add_argument(
        '--category_column',
        required=True,
        help='Column containing the three class labels'
    )

    parser.add_argument(
        '--text_column',
        default='text',
        help='Text column containing the prompts'
    )

    parser.add_argument(
        '--model_name',
        default=DEFAULT_MODEL,
        help='Hugging Face Transformer model name'
    )

    parser.add_argument(
        '--output_dir',
        default=None,
        help='Optional classifier output directory'
    )

    parser.add_argument(
        '--max_length',
        type=int,
        default=256,
        help='Maximum Transformer sequence length'
    )

    parser.add_argument(
        '--epochs',
        type=float,
        default=4,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=2e-5,
        help='Learning rate'
    )

    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=8,
        help='Per-device training batch size'
    )

    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=16,
        help='Per-device evaluation batch size'
    )

    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Gradient accumulation steps'
    )

    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='AdamW weight decay'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    parser.add_argument(
        '--no_class_weights',
        action='store_true',
        help='Disable class-weighted cross entropy'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_dir = args.dataset_dir
    result_dir = args.output_dir or os.path.join(
        dataset_dir, 'classifier_result'
    )
    os.makedirs(result_dir, exist_ok=True)

    print('=' * 70)
    print('Transformer Prompt Classifier')
    print('=' * 70)
    print(f'Dataset:       {dataset_dir}')
    print(f'Model:         {args.model_name}')
    print(f'Text column:   {args.text_column}')
    print(f'Category:      {args.category_column}')
    print(f'Max length:    {args.max_length}')
    print(f'Epochs:        {args.epochs}')
    print(f'Learning rate: {args.learning_rate}')
    print(f'Class weights: {not args.no_class_weights}')
    print(f'Device:        {"CUDA" if torch.cuda.is_available() else "CPU"}')

    data = load_data(dataset_dir)
    data, label2id, id2label = prepare_labels(
        data, args.category_column
    )
    data = prepare_text(data, args.text_column)

    print_distribution(data, id2label)

    with open(
        os.path.join(result_dir, 'label_mapping.json'),
        'w',
        encoding='utf-8'
    ) as f:
        json.dump(
            {
                'label2id': label2id,
                'id2label': {
                    str(k): v for k, v in id2label.items()
                }
            },
            f,
            indent=2
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    datasets = build_datasets(data, args.text_column)
    tokenized = tokenize_datasets(
        datasets,
        tokenizer,
        args.text_column,
        args.max_length
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None
    )

    class_weights = None
    if not args.no_class_weights:
        class_weights = calculate_class_weights(
            tokenized['train'], 3
        )
        print(f'\nClass weights: {class_weights.tolist()}')

    training_args = TrainingArguments(
        output_dir=os.path.join(result_dir, 'checkpoints'),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='steps',
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        greater_is_better=True,
        save_total_limit=2,
        report_to='none',
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        seed=args.seed
    )

    logger = MetricsLogger()

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[logger]
    )

    print('\nStarting training...')
    trainer.train()

    best_model_dir = os.path.join(result_dir, 'best_model')
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    save_training_history(
        logger.history,
        result_dir
    )

    print('\nEvaluating test set...')
    metrics = evaluate_test(
        trainer,
        tokenized['test'],
        id2label,
        data['test'],
        result_dir
    )

    print('\n' + '=' * 70)
    print('TEST RESULTS')
    print('=' * 70)
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Macro Precision:{metrics['macro_precision']:.4f}")
    print(f"Macro Recall:   {metrics['macro_recall']:.4f}")
    print(f"Macro F1:       {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:    {metrics['weighted_f1']:.4f}")

    if metrics['macro_auc_ovr'] is not None:
        print(f"Macro AUC:      {metrics['macro_auc_ovr']:.4f}")
        print(f"Weighted AUC:   {metrics['weighted_auc_ovr']:.4f}")

    print(f'\nResults saved to: {result_dir}')


if __name__ == '__main__':
    main()