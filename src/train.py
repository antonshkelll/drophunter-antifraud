"""
Обучение и оценка модели выявления дропперов.

Схема валидации — out-of-time: учимся на периоде 1 (апрель–май 2026),
проверяемся на периоде 2 (июнь–июль 2026). Это честнее случайного сплита:
в проде модель всегда скорит «будущее», а дропперы во втором периоде в основном новые.

Сравниваем:
  1. Правила (как в типичном rule-based антифроде);
  2. Логистическую регрессию;
  3. Градиентный бустинг (HistGradientBoosting).

Главная метрика — не ROC-AUC, а деньги: сколько ущерба предотвращаем
за вычетом стоимости ложных блокировок и ручной проверки, при ограниченной
мощности команды аналитиков.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA, REPORTS = ROOT / "data", ROOT / "reports"
FIG = REPORTS / "figures"

# --- бизнес-допущения (вынесены отдельно, чтобы их можно было обсуждать) ---
REVIEW_COST = 150          # ₽, ручная проверка одного алерта аналитиком
FP_BLOCK_COST = 2_100      # ₽, ложная блокировка: звонок в поддержку 300 ₽ + 15% риск оттока × LTV 12 000 ₽
SAVED_SHARE = 0.5          # доля входящего «грязного» потока, которую спасаем, поймав дроппера в середине его жизни
CAPACITY = 300             # алертов на период, которые успевает разобрать команда

FEATURES = [
    "age", "account_age_days", "days_since_phone_change", "devices", "has_salary",
    "in_cnt", "in_sum", "in_unique_senders", "in_round_share", "in_night_share",
    "in_new_sender_share", "in_elderly_sender_share", "in_max_amount", "max_daily_senders",
    "out_cnt", "out_sum", "cash_share", "external_share", "out_night_share",
    "passthrough_ratio", "median_min_to_out", "out_within_1h_share",
]


def build_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()
    for t in ("clients", "transfers", "cash"):
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{DATA / t}.parquet')")
    sql = (ROOT / "sql" / "features.sql").read_text(encoding="utf-8")
    periods = [("2026-04-01", "2026-05-31"), ("2026-05-31", "2026-07-30")]
    out = []
    for p_start, p_end in periods:
        df = con.execute(sql, {"p_start": p_start, "p_end": p_end}).df()
        out.append(df)
    return out[0], out[1]


def rules_score(df: pd.DataFrame) -> np.ndarray:
    """Типичные правила: «новый телефон» ИЛИ «транзитный счёт» (много незнакомых → быстрый вывод)."""
    phone = df.days_since_phone_change <= 30
    transit = (
        (df.in_unique_senders >= 5)
        & (df.in_new_sender_share > 0.8)
        & (df.passthrough_ratio > 0.8)
        & (df.median_min_to_out < 120)
    )
    return (phone.astype(int) + 2 * transit.astype(int)).to_numpy(float)


def economics(y: np.ndarray, score: np.ndarray, value: np.ndarray) -> pd.DataFrame:
    """Профит при отправке в работу топ-k клиентов по скору, k = 1..N."""
    order = np.argsort(-score, kind="stable")
    y_s, v_s = y[order], value[order]
    k = np.arange(1, len(y) + 1)
    tp = np.cumsum(y_s)
    fp = k - tp
    saved = np.cumsum(v_s * y_s)
    cost = k * REVIEW_COST + fp * FP_BLOCK_COST
    return pd.DataFrame(dict(k=k, tp=tp, fp=fp, saved=saved, cost=cost, profit=saved - cost,
                             precision=tp / k, recall=tp / max(y.sum(), 1)))


def at_k(econ: pd.DataFrame, k: int) -> dict:
    r = econ.iloc[min(k, len(econ)) - 1]
    return dict(k=int(r.k), precision=round(float(r.precision), 3), recall=round(float(r.recall), 3),
                saved_rub=int(r.saved), cost_rub=int(r.cost), profit_rub=int(r.profit))


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    train, test = build_features()
    Xtr, ytr = train[FEATURES], train.target.to_numpy()
    Xte, yte = test[FEATURES], test.target.to_numpy()
    print(f"train: {len(train):,} клиентов, дропперов {ytr.sum()} ({ytr.mean():.2%})")
    print(f"test:  {len(test):,} клиентов, дропперов {yte.sum()} ({yte.mean():.2%})")

    log1p = FunctionTransformer(np.log1p)
    models = {
        "Правила": None,
        "Логистическая регрессия": make_pipeline(log1p, StandardScaler(),
                                                 LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Градиентный бустинг": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            class_weight="balanced", random_state=0),
    }
    value = test.in_sum.to_numpy() * SAVED_SHARE
    scores, results = {}, {}
    for name, m in models.items():
        if m is None:
            s = rules_score(Xte)
        else:
            m.fit(Xtr, ytr)
            s = m.predict_proba(Xte)[:, 1]
        scores[name] = s
        econ = economics(yte, s, value)
        best = econ.loc[econ.profit.idxmax()]
        results[name] = dict(
            roc_auc=round(roc_auc_score(yte, s), 4),
            pr_auc=round(average_precision_score(yte, s), 4),
            at_capacity=at_k(econ, CAPACITY),
            best_k=at_k(econ, int(best.k)),
        )
        print(f"{name:26s} ROC-AUC={results[name]['roc_auc']:.3f}  PR-AUC={results[name]['pr_auc']:.3f}  "
              f"@{CAPACITY}: P={results[name]['at_capacity']['precision']:.2f} "
              f"R={results[name]['at_capacity']['recall']:.2f} profit={results[name]['at_capacity']['profit_rub']:,} ₽")

    # --- графики ---
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"Правила": "#9aa0a6", "Логистическая регрессия": "#4C78A8", "Градиентный бустинг": "#E45756"}

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(yte, s)
        ax.plot(r, p, label=f"{name} (PR-AUC {results[name]['pr_auc']:.2f})", color=colors[name], lw=2)
    ax.axhline(yte.mean(), ls=":", color="k", lw=1, label=f"случайный ({yte.mean():.1%})")
    ax.set(xlabel="Recall (доля пойманных дропперов)", ylabel="Precision (доля верных алертов)",
           title="Precision–Recall на out-of-time периоде")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "pr_curve.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, s in scores.items():
        econ = economics(yte, s, value).iloc[:2000]
        ax.plot(econ.k, econ.profit / 1e6, label=name, color=colors[name], lw=2)
    ax.axvline(CAPACITY, ls="--", color="k", lw=1)
    ax.text(CAPACITY + 20, ax.get_ylim()[0] + 0.5, f"мощность команды\n{CAPACITY} алертов", fontsize=8)
    ax.set(xlabel="Сколько клиентов отправляем на проверку (топ-k по скору)",
           ylabel="Чистый эффект, млн ₽ за 2 мес.", title="Экономика: профит от порога")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "profit_curve.png"); plt.close(fig)

    gbm = models["Градиентный бустинг"]
    imp = permutation_importance(gbm, Xte, yte, scoring="average_precision", n_repeats=5, random_state=0, n_jobs=-1)
    imp_s = pd.Series(imp.importances_mean, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    imp_s.tail(12).plot.barh(ax=ax, color="#4C78A8")
    ax.set(title="Важность признаков (permutation, падение PR-AUC)", xlabel="ΔPR-AUC")
    fig.tight_layout(); fig.savefig(FIG / "feature_importance.png"); plt.close(fig)

    # где модель ошибается: топ ложных срабатываний по сегментам
    test = test.assign(score=scores["Градиентный бустинг"])
    seg = pd.read_parquet(DATA / "clients.parquet", columns=["client_id", "segment", "sophisticated"])
    top = test.nlargest(CAPACITY, "score").merge(seg, on="client_id")
    fp_by_segment = top[top.target == 0].segment.value_counts().to_dict()
    caught = test[test.target == 1].merge(seg, on="client_id")
    caught["in_top"] = caught.client_id.isin(top.client_id)
    recall_by_type = caught.groupby("sophisticated").in_top.mean().round(3)
    recall_by_type.index = recall_by_type.index.map({False: "простые", True: "продвинутые"})

    summary = dict(
        train=dict(clients=len(train), droppers=int(ytr.sum())),
        test=dict(clients=len(test), droppers=int(yte.sum())),
        assumptions=dict(review_cost=REVIEW_COST, fp_block_cost=FP_BLOCK_COST,
                         saved_share=SAVED_SHARE, capacity=CAPACITY),
        models=results,
        top_features=imp_s.sort_values(ascending=False).head(8).round(4).to_dict(),
        gbm_false_positives_by_segment=fp_by_segment,
        gbm_recall_at_capacity_by_dropper_type=recall_by_type.to_dict(),
    )
    (REPORTS / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
