import os
import re
import argparse
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from transformers import AutoTokenizer


plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11})

TOKENIZER_NAME = 'answerdotai/ModernBERT-base'

try:
    hf_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
except Exception as e:
    print(f"Warning: Failed to load {TOKENIZER_NAME}. Falling back to gpt2. Error: {e}")
    hf_tokenizer = AutoTokenizer.from_pretrained('gpt2')


def get_token_counts(texts, batch_size=2048):
    """Return HF subword-token counts using batched tokenization."""
    index = texts.index if isinstance(texts, pd.Series) else None
    texts = texts.fillna('').astype(str).tolist() if isinstance(texts, pd.Series) else [
        str(x) if x is not None else '' for x in texts
    ]

    counts = []
    for i in range(0, len(texts), batch_size):
        result = hf_tokenizer(
            texts[i:i + batch_size],
            add_special_tokens=False,
            truncation=False,
            return_length=True
        )
        counts.extend(result['length'])

    return pd.Series(counts, index=index, dtype='int32')


def tokenize_subwords(text):
    """Convert text into cleaned HF subword tokens for TF-IDF."""
    if not isinstance(text, str) or not text.strip():
        return []

    ids = hf_tokenizer.encode(text, add_special_tokens=False, truncation=False)
    tokens = hf_tokenizer.convert_ids_to_tokens(ids)
    return [re.sub(r'^[Ġ##]', '', token) for token in tokens if token.strip()]


def get_token_series(df, text_column='text'):
    """Use cached token counts when available."""
    return df['token_count'] if 'token_count' in df.columns else get_token_counts(df[text_column])


def clean_text_content(text):
    """Remove control characters and normalize horizontal whitespace."""
    if not isinstance(text, str):
        return ''

    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    return re.sub(r'[ \t]+', ' ', text).strip()


def clean_df(df, text_column, binary_column):
    """Clean text, remove duplicates, and normalize selected columns."""
    cleaned = df.copy()
    initial_len = len(cleaned)

    if text_column in cleaned.columns:
        cleaned[text_column] = cleaned[text_column].astype(str).map(clean_text_content)
        cleaned = cleaned[
            (cleaned[text_column].str.len() > 0) &
            (cleaned[text_column].str.lower() != 'nan')
        ].drop_duplicates(subset=text_column)
    else:
        cleaned = cleaned.drop_duplicates()

    if binary_column in cleaned.columns:
        cleaned[binary_column] = cleaned[binary_column].astype(int)

    if 'augmented' in cleaned.columns:
        cleaned['augmented'] = cleaned['augmented'].astype(bool)

    print(f"Cleaning complete. Removed {initial_len - len(cleaned)} invalid/duplicate rows.")
    return cleaned


def save_figure(fig, filename, message):
    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"{message} {filename}")


def plot_token_statistics_by_label(
    df, text_column, binary_column, name, result_dir, prefix=''
):
    """Plot token distributions and statistical summaries by class."""
    if text_column not in df.columns or binary_column not in df.columns:
        return

    token_count = get_token_series(df, text_column)
    summary = token_count.groupby(df[binary_column]).agg(
        Count='count',
        Mean='mean',
        Median='median',
        Mode=lambda x: stats.mode(x, keepdims=True).mode[0] if len(x) else 0,
        Std='std',
        Min='min',
        Q1=lambda x: x.quantile(.25),
        Q3=lambda x: x.quantile(.75),
        Max='max'
    ).reset_index()

    print(
        f"\n--- BPE Subword Token Statistics [{TOKENIZER_NAME}] "
        f"({name}) ---\n{summary.to_string(index=False)}"
    )

    plot_data = pd.DataFrame({
        binary_column: df[binary_column].values,
        'token_count': token_count.values
    })

    fig, (ax_plot, ax_table) = plt.subplots(
        2, 1, figsize=(12, 8),
        gridspec_kw={'height_ratios': [2.5, 1]}
    )

    sns.violinplot(
        data=plot_data, x=binary_column, y='token_count',
        hue=binary_column, ax=ax_plot, palette='Set2',
        inner=None, alpha=.3, legend=False
    )
    sns.boxplot(
        data=plot_data, x=binary_column, y='token_count',
        hue=binary_column, ax=ax_plot, palette='Set2',
        width=.25, boxprops={'alpha': .85}, showmeans=True,
        meanprops={
            'marker': 'o', 'markerfacecolor': 'white',
            'markeredgecolor': 'black', 'markersize': '6'
        },
        legend=False
    )

    ax_plot.set_title(
        f'[{name}] Subword Token Distribution by Label ({TOKENIZER_NAME})',
        fontsize=14, fontweight='bold', pad=15
    )
    ax_plot.set(xlabel='Label', ylabel='Subword Token Count')
    ax_plot.grid(True, axis='y', ls='--', alpha=.5)
    sns.despine(ax=ax_plot)

    headers = [
        'Label', 'Count (N)', 'Mean', 'Median', 'Mode',
        'Std Dev', 'Min', 'Q1', 'Q3', 'Max'
    ]
    table_data = [
        [
            f"Class {int(row[binary_column])}",
            f"{int(row.Count):,}", f"{row.Mean:.1f}",
            f"{row.Median:.1f}", f"{row.Mode:.1f}",
            f"{row.Std:.1f}", f"{int(row.Min)}",
            f"{row.Q1:.1f}", f"{row.Q3:.1f}", f"{int(row.Max)}"
        ]
        for _, row in summary.iterrows()
    ]

    ax_table.axis('off')
    table = ax_table.table(
        cellText=table_data, colLabels=headers,
        cellLoc='center', loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.4)

    for (i, _), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#4C72B0')
            cell.set_text_props(color='white', fontweight='bold')
        elif i % 2 == 0:
            cell.set_facecolor('#F2F2F2')

    filename = os.path.join(
        result_dir,
        f"{prefix}{name.lower().replace(' ', '_')}_subword_token_stats_by_label.png"
    )
    save_figure(fig, filename, "Saved token statistics plot to")


def generate_per_class_wordclouds(
    df, text_column, binary_column, name, result_dir, prefix=''
):
    """Generate TF-IDF and one-vs-rest Chi2 subword word clouds."""
    if text_column not in df.columns or binary_column not in df.columns:
        return

    print(f"\nGenerating per-class subword word clouds for {name}...")

    texts = df[text_column].astype(str).tolist()
    labels = df[binary_column].to_numpy()
    classes = np.sort(df[binary_column].unique())

    vectorizer = TfidfVectorizer(
        max_features=5000,
        tokenizer=tokenize_subwords,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(texts)
    features = vectorizer.get_feature_names_out()

    fig, axes = plt.subplots(
        len(classes), 2, figsize=(14, 6 * len(classes))
    )
    axes = np.atleast_2d(axes)

    for i, cls in enumerate(classes):
        mask = labels == cls
        tfidf_scores = np.asarray(X[mask].mean(axis=0)).ravel()
        tfidf_dict = dict(zip(features, tfidf_scores))

        chi2_scores, _ = chi2(X, mask.astype(np.int8))
        chi2_dict = dict(zip(features, chi2_scores))

        tfidf_wc = WordCloud(
            width=800, height=400, background_color='white',
            colormap='Blues' if cls == 1 else 'Purples',
            max_words=100, regexp=r'\S+'
        ).generate_from_frequencies(tfidf_dict)

        chi2_wc = WordCloud(
            width=800, height=400, background_color='white',
            colormap='Reds' if cls == 1 else 'Oranges',
            max_words=100, regexp=r'\S+'
        ).generate_from_frequencies(chi2_dict)

        for ax, cloud, title in [
            (axes[i, 0], tfidf_wc, 'Mean TF-IDF Subwords'),
            (axes[i, 1], chi2_wc, 'One-vs-Rest Chi2 Subwords')
        ]:
            ax.imshow(cloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(
                f'[{name}] Class {cls} - {title}',
                fontsize=12, fontweight='bold', pad=10
            )

    filename = os.path.join(
        result_dir,
        f"{prefix}{name.lower().replace(' ', '_')}_per_class_wordclouds.png"
    )
    save_figure(fig, filename, "Saved per-class word cloud plot to")


def plot_category_distribution(
    df, category_column, name, result_dir, prefix=''
):
    """Plot category frequencies on a logarithmic scale."""
    if category_column not in df.columns:
        return

    counts = df[category_column].value_counts()
    fig, ax = plt.subplots(figsize=(10, max(8, len(counts) * .35)))

    bars = ax.barh(counts.index, counts.values, color='maroon')
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    for bar in bars:
        width = bar.get_width()
        y_pos = bar.get_y() + bar.get_height() / 2
        ax.text(
            width * 1.15, y_pos, f'{int(width):,}',
            va='center', ha='left', fontsize=8,
            color='black', weight='bold'
        )

    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax * 3)

    ax.set_title(
        f'[{name}] Category Distribution (Log Scale)',
        fontsize=14, fontweight='bold', pad=15
    )
    ax.set(xlabel='Count (Log Scale)', ylabel='Category')
    ax.grid(True, which='both', axis='x', ls='--', alpha=.4)

    filename = os.path.join(
        result_dir,
        f"{prefix}{name.lower().replace(' ', '_')}_category_distribution_log.png"
    )
    save_figure(fig, filename, "Saved log-scaled category distribution plot to")


def perform_eda(
    df, name, text_column, binary_column, category_column
):
    """Print core dataset statistics."""
    print(f"\n{'=' * 20} EDA REPORT: {name} {'=' * 20}")
    print(f"Tokenizer Model: {TOKENIZER_NAME}")
    print(f"Total Rows: {len(df):,}")
    print(f"Total Columns: {df.shape[1]}")

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    if text_column in df.columns:
        total_tokens = int(get_token_series(df, text_column).sum())
        print("\n--- Subword Tokenization Insights ---")
        print(f"Total Corpus Subwords: {total_tokens:,}")
        print(f"Tokenizer Vocabulary Size: {hf_tokenizer.vocab_size:,}")

    if binary_column in df.columns:
        print("\n--- Label Distribution ---")
        print(df[binary_column].value_counts(normalize=True) * 100)

    if category_column in df.columns:
        print("\n--- Category Breakdown ---")
        print(df[category_column].value_counts())

    if 'augmented' in df.columns:
        print("\n--- Augmentation Ratio ---")
        print(df['augmented'].value_counts(normalize=True) * 100)


def plot_statistics(
    df, name, result_dir, text_column, binary_column, category_column, prefix=''
):
    """Create label, token-length, and category summary plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if binary_column in df.columns:
        sns.countplot(
            data=df, x=binary_column, hue=binary_column,
            ax=axes[0], palette='Blues_d', legend=False
        )
        axes[0].set_title(
            f'[{name}] Label Distribution',
            fontsize=12, fontweight='bold'
        )
        axes[0].set(xlabel='Label', ylabel='Count')

        for p in axes[0].patches:
            axes[0].annotate(
                f'{int(p.get_height()):,}',
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='center',
                xytext=(0, 5), textcoords='offset points'
            )

    if text_column in df.columns:
        sns.histplot(
            get_token_series(df, text_column), bins=30, kde=True,
            ax=axes[1], color='teal'
        )
        axes[1].set_title(
            f'[{name}] Subword Token Distribution',
            fontsize=12, fontweight='bold'
        )
        axes[1].set(xlabel='Subword Token Count', ylabel='Frequency')

    if category_column in df.columns:
        top = (
            df[category_column]
            .value_counts()
            .head(5)
            .rename_axis(category_column)
            .reset_index(name='count')
        )

        sns.barplot(
            data=top, x='count', y=category_column,
            hue=category_column, ax=axes[2],
            palette='viridis', legend=False
        )
        axes[2].set_title(
            f'[{name}] Top Categories',
            fontsize=12, fontweight='bold'
        )
        axes[2].set(xlabel='Count', ylabel='Category')

    filename = os.path.join(
        result_dir,
        f"{prefix}{name.lower().replace(' ', '_')}_eda_plots.png"
    )
    save_figure(fig, filename, "Saved EDA plots to")


def process_dataset(dataset_dir, text_column, binary_column, category_column):
    """Run the complete EDA, cleaning, plotting, and saving pipeline."""
    raw_data_dir = os.path.join(dataset_dir, 'raw_data_dir')
    clean_data_dir = os.path.join(dataset_dir, 'clean_data_dir')
    os.makedirs(clean_data_dir, exist_ok=True)

    file_names = {
        'Train Set': 'train-00000-of-00001.parquet',
        'Test Set': 'test-00000-of-00001.parquet',
        'Validation Set': 'validation-00000-of-00001.parquet'
    }

    result_names = {
        'Train Set': 'train_result',
        'Test Set': 'test_result',
        'Validation Set': 'validation_result',
        'Full Dataset': 'full_result'
    }

    datasets = {
        name: pd.read_parquet(os.path.join(raw_data_dir, filename))
        for name, filename in file_names.items()
    }
    datasets['Full Dataset'] = pd.concat(
        datasets.values(), ignore_index=True
    )

    output_names = {
        **file_names,
        'Full Dataset': 'full_dataset.parquet'
    }

    for name, dataset in datasets.items():
        result_dir = os.path.join(dataset_dir, result_names[name])
        os.makedirs(result_dir, exist_ok=True)

        print(f"\n{'=' * 15} Processing: {name} {'=' * 15}")

        if text_column in dataset.columns:
            dataset['token_count'] = get_token_counts(dataset[text_column])

        perform_eda(dataset, f"Raw {name}", text_column, binary_column, category_column)
        plot_statistics(
            dataset, name, result_dir,
            text_column, binary_column, category_column
        )
        plot_token_statistics_by_label(
            dataset, text_column, binary_column, name, result_dir
        )
        plot_category_distribution(
            dataset, category_column, name, result_dir
        )
        generate_per_class_wordclouds(
            dataset, text_column, binary_column, name, result_dir
        )

        cleaned_df = clean_df(dataset, text_column, binary_column)

        if text_column in cleaned_df.columns:
            cleaned_df['token_count'] = get_token_counts(cleaned_df[text_column])

        perform_eda(
            cleaned_df, f"Clean {name}",
            text_column, binary_column, category_column
        )
        plot_statistics(
            cleaned_df, name, result_dir,
            text_column, binary_column, category_column, prefix='clean_'
        )
        plot_token_statistics_by_label(
            cleaned_df, text_column, binary_column, name,
            result_dir, prefix='clean_'
        )
        plot_category_distribution(
            cleaned_df, category_column, name,
            result_dir, prefix='clean_'
        )
        generate_per_class_wordclouds(
            cleaned_df, text_column, binary_column, name,
            result_dir, prefix='clean_'
        )

        save_path = os.path.join(clean_data_dir, output_names[name])
        cleaned_df.to_parquet(save_path, index=False)
        print(f"Saved cleaned {name} parquet to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='EDA, cleaning, and visualization for parquet datasets.'
    )
    parser.add_argument(
        '--dataset_dir', required=True,
        help='Dataset directory, e.g. dataset_1'
    )
    parser.add_argument(
        '--binary_column', required=True,
        help='Binary label column name, e.g. label, label_binary'
    )
    parser.add_argument(
        '--category_column', required=True,
        help='Category column name, e.g. category, label_category'
    )
    parser.add_argument(
        '--text_column',
        default='text',
        help='Text column containing the prompts'
    )

    args = parser.parse_args()

    process_dataset(
        args.dataset_dir,
        args.text_column,
        args.binary_column,
        args.category_column
    )


if __name__ == '__main__':
    main()