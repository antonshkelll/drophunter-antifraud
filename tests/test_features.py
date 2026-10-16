"""SQL-признаки на маленьком примере, где правильные значения считаются вручную."""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train import economics  # noqa: E402

SQL = (ROOT / "sql" / "features.sql").read_text(encoding="utf-8")


@pytest.fixture()
def con():
    clients = pd.DataFrame({
        "client_id": [1, 2, 3, 4], "age": [20, 60, 61, 35], "account_age_at_start": [10, 900, 900, 500],
        "phone_change_day": [-9999, -9999, -9999, -9999], "devices": [3, 1, 1, 1], "has_salary": [False, True, True, True],
        "is_dropper": [True, False, False, False], "confirmed": [True, False, False, False],
        "act_start": [0, -1, -1, -1], "act_len": [30, 0, 0, 0],
    })
    t0 = pd.Timestamp("2026-04-05 12:00")
    transfers = pd.DataFrame({
        "transfer_id": range(6),
        # клиент 1 получает 3 перевода от незнакомых людей и через 10 минут переводит 95% дальше
        "sender":   [2, 3, 4, 1, 1, 1],
        "receiver": [1, 1, 1, -1, -1, -1],
        "ts": [t0, t0 + pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=2),
               t0 + pd.Timedelta(minutes=10), t0 + pd.Timedelta(minutes=70), t0 + pd.Timedelta(minutes=130)],
        "amount": [10000.0, 20000.0, 30000.0, 9500.0, 19000.0, 28500.0],
        "is_fraud": [True] * 6,
    })
    cash = pd.DataFrame({"client_id": pd.Series([], dtype=int), "ts": pd.Series([], dtype="datetime64[ns]"),
                         "amount": pd.Series([], dtype=float)})
    c = duckdb.connect()
    for name, df in (("clients", clients), ("transfers", transfers), ("cash", cash)):
        c.register(name, df)
    return c


def test_dropper_features(con):
    f = con.execute(SQL, {"p_start": "2026-04-01", "p_end": "2026-05-31"}).df().set_index("client_id")
    d = f.loc[1]
    assert d.in_cnt == 3 and d.in_unique_senders == 3
    assert d.in_new_sender_share == 1.0          # раньше эти люди клиенту не переводили
    assert d.passthrough_ratio == pytest.approx(0.95)
    assert d.median_min_to_out == 10
    assert d.out_within_1h_share == 1.0
    assert d.in_round_share == 1.0
    assert d.external_share == 1.0
    assert d.target == 1


def test_only_clients_with_inflow_are_scored(con):
    f = con.execute(SQL, {"p_start": "2026-04-01", "p_end": "2026-05-31"}).df()
    assert list(f.client_id) == [1]


def test_economics_profit():
    y = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.7, 0.1])
    value = np.array([100_000.0, 0, 50_000.0, 0])
    e = economics(y, score, value)
    # топ-1: поймали дроппера, одна проверка
    assert e.profit.iloc[0] == 100_000 - 150
    # топ-2: добавился ложный алерт
    assert e.profit.iloc[1] == 100_000 - 2 * 150 - 2_100
    assert e.recall.iloc[2] == 1.0
