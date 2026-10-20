"""
Мониторинг дрейфа признаков: PSI между периодом обучения и периодом скоринга.

Мошенники адаптируются: меняют суммы, время переводов, схемы обналичивания. Если распределение
признака заметно уехало, модель начинает ошибаться раньше, чем это видно по метрикам — разметка
(жалобы жертв) приходит с задержкой. Поэтому PSI считается по каждому признаку на каждом новом
периоде, до того как появятся подтверждённые дропперы.

PSI = Σ (доля_новая − доля_базовая) × ln(доля_новая / доля_базовая) по бинам базового периода.
Общепринятые пороги: < 0,1 — стабильно, 0,1–0,25 — заметный сдвиг, > 0,25 — сильный дрейф,
модель стоит перепроверить.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from train import FEATURES, FIG, REPORTS, build_features, plt

STABLE, SHIFT = 0.10, 0.25
EPS = 1e-4


def psi(base: np.ndarray, new: np.ndarray, bins: int = 10) -> float:
    """PSI с бинами по квантилям базового периода. Пропуски игнорируются."""
    base = np.asarray(base, dtype=float)
    new = np.asarray(new, dtype=float)
    base, new = base[~np.isnan(base)], new[~np.isnan(new)]
    if len(base) == 0 or len(new) == 0:
        return float("nan")
    edges = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:                     # константный признак: сравниваем долю того же значения
        same_base, same_new = 1.0, float(np.mean(new == base[0]))
        b, n = np.array([same_base, 1 - same_base]) + EPS, np.array([same_new, 1 - same_new]) + EPS
        return float(np.sum((n - b) * np.log(n / b)))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.histogram(base, edges)[0] / len(base) + EPS
    n = np.histogram(new, edges)[0] / len(new) + EPS
    return float(np.sum((n - b) * np.log(n / b)))


def status(value: float) -> str:
    if value < STABLE:
        return "стабильно"
    return "заметный сдвиг" if value < SHIFT else "сильный дрейф"


def drift_report(base: pd.DataFrame, new: pd.DataFrame, features: list[str] = FEATURES) -> pd.DataFrame:
    rows = [{"feature": f, "psi": round(psi(base[f], new[f]), 4)} for f in features]
    df = pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)
    df["status"] = df.psi.map(status)
    return df


def main() -> dict:
    train, test = build_features()
    rep = drift_report(train, test)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = ["#d62728" if v >= SHIFT else "#ff7f0e" if v >= STABLE else "#4C78A8" for v in rep.psi]
    ax.barh(rep.feature[::-1], rep.psi[::-1], color=colors[::-1])
    for x, name in [(STABLE, "0,10"), (SHIFT, "0,25")]:
        ax.axvline(x, color="k", ls="--", lw=0.8)
        ax.text(x, len(rep) - 0.4, name, ha="center", fontsize=8)
    ax.set(title="Дрейф признаков: апрель–май → июнь–июль (PSI)", xlabel="PSI")
    fig.tight_layout(); fig.savefig(FIG / "psi.png", dpi=130); plt.close(fig)

    summary = {
        "base_period": "2026-04-01 — 2026-05-31",
        "new_period": "2026-05-31 — 2026-07-30",
        "thresholds": {"stable": STABLE, "shift": SHIFT},
        "counts": rep.status.value_counts().to_dict(),
        "features": rep.to_dict("records"),
    }
    (REPORTS / "psi.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(rep.to_string(index=False))
    return summary


if __name__ == "__main__":
    main()
