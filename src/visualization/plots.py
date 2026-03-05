"""
plots.py
========
Графики для диплома — все 4 типа из тех-плана.

1. Reward over training steps (line plot)
2. F1-score bar chart (сравнение методов)
3. Loss curves PPO vs GRPO
4. Radar chart: качество / скорость / память / стабильность

Зависимости: matplotlib, plotly, kaleido
"""

import os
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np

# Глобальный стиль
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi":   150,
    "savefig.dpi":  300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "baseline":    "#6c757d",
    "grpo":        "#2196F3",
    "ppo":         "#FF5722",
    "dapo":        "#4CAF50",
    "lambda_grpo": "#9C27B0",
}

OUTPUT_DIR = "./outputs/plots"


# ──────────────────────────────────────────────────────────────────────────────
# 1. Reward / Loss over training steps
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    history: Dict[str, Dict[str, List[float]]],
    metric: str = "reward/mean",
    title: str = None,
    save_name: str = "training_curves.png",
) -> str:
    """
    Строит кривые обучения для нескольких методов.

    Args:
        history: dict вида {method_name: {metric_name: [values]}}
        metric:  Название метрики (reward/mean, train/loss, eval/f1)
        title:   Заголовок графика
        save_name: Имя файла для сохранения

    Returns:
        Путь к сохранённому файлу
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for method, data in history.items():
        if metric in data:
            values = data[metric]
            steps = list(range(len(values)))
            color = COLORS.get(method, "#333333")
            ax.plot(steps, values, label=method.upper(), color=color, linewidth=2)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} over training")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    path = _save_fig(fig, save_name)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# 2. F1-score bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_f1_comparison(
    results: Dict[str, float],
    save_name: str = "f1_comparison.png",
) -> str:
    """
    Bar chart сравнения F1-score по методам.

    Args:
        results: dict {method_name: f1_score}
        save_name: Имя файла

    Returns:
        Путь к сохранённому файлу
    """
    methods = list(results.keys())
    f1_scores = list(results.values())
    colors = [COLORS.get(m, "#333333") for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [m.upper() for m in methods],
        f1_scores,
        color=colors,
        width=0.6,
        edgecolor="white",
        linewidth=1.5,
    )

    # Подписи значений
    for bar, val in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    best_method = max(results, key=results.get)
    ax.set_title(f"F1-score Comparison (лучший: {best_method.upper()})")
    ax.set_ylabel("F1-score (weighted)")
    ax.set_ylim(0, max(f1_scores) * 1.15)
    ax.grid(True, axis="y", alpha=0.3)

    path = _save_fig(fig, save_name)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# 3. Radar chart: качество / скорость / память / стабильность
# ──────────────────────────────────────────────────────────────────────────────

def plot_radar_chart(
    scores: Dict[str, Dict[str, float]],
    save_name: str = "radar_chart.png",
) -> str:
    """
    Radar chart для сравнения методов по 4 измерениям.

    Args:
        scores: dict вида {
            method_name: {
                "quality":     0.0–1.0,
                "speed":       0.0–1.0,
                "memory":      0.0–1.0,  (инвертированная — меньше = лучше)
                "stability":   0.0–1.0,
            }
        }
    """
    categories = ["Качество (F1)", "Скорость", "Экономия памяти", "Стабильность"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # замыкаем

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for method, vals in scores.items():
        values = [
            vals.get("quality",   0),
            vals.get("speed",     0),
            vals.get("memory",    0),
            vals.get("stability", 0),
        ]
        values += values[:1]
        color = COLORS.get(method, "#333333")
        ax.plot(angles, values, "o-", linewidth=2, label=method.upper(), color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=8)
    ax.set_title("Сравнение методов", size=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    ax.grid(True, alpha=0.3)

    path = _save_fig(fig, save_name)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# 4. Plotly интерактивный radar (для вставки в диплом как HTML)
# ──────────────────────────────────────────────────────────────────────────────

def plot_radar_plotly(
    scores: Dict[str, Dict[str, float]],
    save_name: str = "radar_interactive.html",
) -> str:
    """Интерактивный radar chart через Plotly (HTML-файл)."""
    try:
        import plotly.graph_objects as go

        categories = ["Качество", "Скорость", "Память", "Стабильность"]
        fig = go.Figure()

        for method, vals in scores.items():
            values = [
                vals.get("quality", 0),
                vals.get("speed", 0),
                vals.get("memory", 0),
                vals.get("stability", 0),
            ]
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=method.upper(),
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Сравнение методов RL (интерактивный)",
            showlegend=True,
        )

        path = os.path.join(OUTPUT_DIR, save_name)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig.write_html(path)
        return path
    except ImportError:
        print("plotly не установлен: pip install plotly")
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные
# ──────────────────────────────────────────────────────────────────────────────

def _save_fig(fig, filename: str) -> str:
    """Сохраняет matplotlib figure в OUTPUT_DIR."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"График сохранён: {path}")
    return path


def plot_all_from_results(
    all_results: List[Dict[str, Any]],
    training_history: Optional[Dict] = None,
) -> List[str]:
    """
    Удобная функция — строит все графики из результатов evaluate_checkpoint.

    Returns:
        Список путей к сохранённым файлам
    """
    saved_paths = []

    # F1 comparison
    f1_scores = {r["method"]: r["f1_weighted"] for r in all_results}
    saved_paths.append(plot_f1_comparison(f1_scores))

    # Training curves (если есть история)
    if training_history:
        saved_paths.append(plot_training_curves(training_history, metric="reward/mean"))
        saved_paths.append(plot_training_curves(
            training_history, metric="train/loss", save_name="loss_curves.png"
        ))

    return saved_paths
