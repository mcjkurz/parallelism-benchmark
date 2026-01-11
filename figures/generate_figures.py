"""
Generate publication-quality figures and comparison tables from evaluation results.

Usage:
    python figures/generate_figures.py
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RESULTS_FILE = os.path.join(PROJECT_ROOT, "results", "evaluation_results.json")
OUTPUT_DIR = SCRIPT_DIR  # Output to figures/ directory where this script lives


def load_results():
    """Load evaluation results from JSON."""
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: Results file not found: {RESULTS_FILE}")
        print("Run 'python scripts/run_trials.py' first to generate results.")
        sys.exit(1)
    
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def configure_matplotlib():
    """Configure matplotlib for publication-quality output."""
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'figure.figsize': (10, 6),
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica Now', 'Avenir Next', 'Avenir', 'Helvetica', 'Arial', 'sans-serif'],
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
        'lines.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
    })


def extract_trial_values(results, model_key, metric):
    """Extract metric values from trials."""
    trials = results.get("trials", [])
    values = []
    for trial in trials:
        if model_key in trial and metric in trial[model_key]:
            values.append(trial[model_key][metric])
    return values


def get_target_colors():
    """Return color scheme grouped by evaluation target."""
    return {
        # Character target (teal/blue)
        'char': '#2E86AB',
        # Couplet target (purple shades)
        'couplet': '#8E44AD',
        'char_to_couplet': '#A569BD',
        # Poem1 target (orange/amber shades) - poem1 is anchor
        'poem1': '#D35400',
        'couplet_to_poem': '#E67E22',
        'char_to_poem': '#F39C12',
        # Poem4 target (green shades)
        'poem4': '#16A085',
        'poem4_to_poem': '#05BC9C',
    }


def generate_boxplot(results, output_dir):
    """Generate box plot grouped by target: char | couplet | poem1 | poem4."""
    # Grouped: Character | Couplet group | Poem-1 group | Poem-4 group
    models = [
        "char",
        "couplet", "char_to_couplet",
        "poem1", "couplet_to_poem", "char_to_poem",
        "poem4", "poem4_to_poem"
    ]
    labels = [
        "Character",
        "Couplet", "Char→Couplet",
        "Poem-1", "Couplet→Poem-1", "Char→Poem-1",
        "Poem-4", "Poem-4→Poem-1"
    ]
    target_colors = get_target_colors()
    
    data = []
    valid_labels = []
    valid_colors = []
    
    for i, model in enumerate(models):
        values = extract_trial_values(results, model, "f1")
        if len(values) > 0:
            data.append(values)
            valid_labels.append(labels[i])
            valid_colors.append(target_colors[model])
    
    if not data:
        print("  Warning: No data for box plot")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # First, draw scatter points for each trial with horizontal jitter (behind boxes)
    np.random.seed(42)  # For reproducibility
    for i, (values, color) in enumerate(zip(data, valid_colors)):
        x_pos = i + 1  # Boxplot positions are 1-indexed
        jitter = np.random.uniform(-0.15, 0.15, size=len(values))
        ax.scatter(x_pos + jitter, values, color=color, alpha=0.55, s=30, 
                   edgecolors='none', zorder=2)
    
    # Draw boxplot on top with transparent boxes
    bp = ax.boxplot(data, tick_labels=valid_labels, patch_artist=True,
                    medianprops={'color': 'black', 'linewidth': 2},
                    whiskerprops={'linewidth': 1.5, 'color': '#333333'},
                    capprops={'linewidth': 1.5, 'color': '#333333'},
                    showfliers=False,  # Hide outlier markers since we show all points
                    zorder=3)
    
    for patch, color in zip(bp['boxes'], valid_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)  # Transparent boxes
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_xlabel('Model / Inference', fontweight='bold')
    ax.set_ylim(0.5, 1.02)
    
    # Add dividers between target groups (positions are 1-indexed for boxplot, more visible)
    ax.axvline(x=1.5, color='#555555', linestyle='--', linewidth=2, alpha=0.8)  # After char
    ax.axvline(x=3.5, color='#555555', linestyle='--', linewidth=2, alpha=0.8)  # After couplet group
    ax.axvline(x=6.5, color='#555555', linestyle='--', linewidth=2, alpha=0.8)  # After poem1 group
    
    # Mean markers on top
    means = [np.mean(d) for d in data]
    ax.scatter(range(1, len(means) + 1), means, color='white', s=60, zorder=5,
               marker='D', edgecolors='black', linewidth=1.2, label='Mean')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'model_f1_boxplot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def print_comparison_table(results):
    """Print comprehensive comparison table (grouped by target)."""
    # Grouped: Character | Couplet group | Poem-1 group | Poem-4 group
    models = [
        "char",
        "couplet", "char_to_couplet",
        "poem1", "couplet_to_poem", "char_to_poem",
        "poem4", "poem4_to_poem"
    ]
    display_names = {
        "char": "Character",
        "couplet": "Couplet",
        "poem4": "Poem-4",
        "poem1": "Poem-1",
        "char_to_couplet": "Char → Couplet",
        "couplet_to_poem": "Couplet → Poem1",
        "poem4_to_poem": "Poem4 → Poem1",
        "char_to_poem": "Char → Poem1",
    }
    metrics = ["accuracy", "precision", "recall", "f1"]
    
    print()
    print("=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)
    
    header = f"{'Model':<20}"
    for metric in metrics:
        header += f" {metric.capitalize():>18}"
    print(header)
    print("-" * 100)
    
    for model in models:
        row = f"{display_names[model]:<20}"
        for metric in metrics:
            values = extract_trial_values(results, model, metric)
            mean = np.mean(values) if values else 0
            std = np.std(values) if values else 0
            row += f" {mean:.4f}±{std:.4f}"
        print(row)
    
    print("=" * 100)
    
    # LaTeX table
    print()
    print("LATEX TABLE")
    print("-" * 100)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Model Performance Comparison}")
    print(r"\begin{tabular}{l" + "c" * len(metrics) + "}")
    print(r"\hline")
    print(r"Model & " + " & ".join([m.capitalize() for m in metrics]) + r" \\")
    print(r"\hline")
    
    for model in models:
        row_parts = [display_names[model]]
        for metric in metrics:
            values = extract_trial_values(results, model, metric)
            if values:
                mean = np.mean(values)
                std = np.std(values)
                row_parts.append(f"${mean:.3f} \\pm {std:.3f}$")
            else:
                row_parts.append("N/A")
        print(" & ".join(row_parts) + r" \\")
    
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


def main():
    print("=" * 60)
    print("Generating Figures")
    print("=" * 60)
    print()
    
    print(f"Loading results from: {RESULTS_FILE}")
    results = load_results()
    print(f"Loaded {results.get('num_trials', 0)} trials")
    
    configure_matplotlib()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print()
    print("Generating figures...")
    generate_boxplot(results, OUTPUT_DIR)
    
    print_comparison_table(results)
    
    print()
    print("=" * 60)
    print(f"Done! Figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
