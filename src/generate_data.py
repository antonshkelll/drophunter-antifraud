"""
Генератор синтетических данных P2P-переводов банка с дропперами.

Почему синтетика: реальные транзакции — банковская тайна, их нет в открытом доступе.
Поэтому генератор воспроизводит паттерны, которые описывают ЦБ и банки:
  * дроппер получает деньги от многих незнакомых отправителей (fan-in)
    и за минуты-часы выводит почти всё дальше (pass-through), часто наличными;
  * перед схемой у дроппера часто меняется номер телефона (признак ЦБ с конца 2025);
  * карты «свежие» или принадлежат студентам, которых завербовали;
  * «продвинутые» дропперы маскируются: держат деньги дольше, суммы не круглые.

Чтобы задача не была игрушечной, в данных есть «трудные» легитимные клиенты:
  * самозанятые/продавцы — тоже получают от многих незнакомых людей и выводят деньги;
  * студенты, получающие переводы от родителей;
  * клиенты, недавно сменившие телефон по бытовым причинам.

Выход: data/clients.parquet, data/transfers.parquet, data/cash.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DAYS = 120                     # два периода по 60 дней: обучение и out-of-time тест
START = pd.Timestamp("2026-04-01")


def make_clients(rng: np.random.Generator, n: int) -> pd.DataFrame:
    ids = np.arange(n)
    age = np.clip(rng.normal(38, 13, n), 18, 85).round().astype(int)
    # стаж счёта в днях к началу наблюдения
    account_age = rng.gamma(2.0, 700, n).astype(int) + 30
    segment = rng.choice(["regular", "seller", "student", "p2p_trader"], n, p=[0.894, 0.035, 0.065, 0.006])
    age[segment == "student"] = rng.integers(18, 23, (segment == "student").sum())

    is_dropper = np.zeros(n, bool)
    n_drop = int(n * 0.012)
    # дропперов чаще вербуют среди молодых
    w = np.where(age < 25, 4.0, np.where(age < 35, 2.0, 0.5))
    drop_ids = rng.choice(ids, n_drop, replace=False, p=w / w.sum())
    is_dropper[drop_ids] = True

    sophisticated = np.zeros(n, bool)
    sophisticated[drop_ids] = rng.random(n_drop) < 0.50
    # метка — это «подтверждённый» дроппер (жалоба жертвы / база ФинЦЕРТ).
    # ~15% реальных дропперов так и не подтверждаются: в данных они выглядят как обычные клиенты.
    confirmed = np.zeros(n, bool)
    confirmed[drop_ids] = rng.random(n_drop) < 0.85

    # окно активности дроппера: старт равномерно по 120 дням, «живёт» 7–35 дней
    act_start = np.full(n, -1)
    act_len = np.zeros(n, int)
    act_start[drop_ids] = rng.integers(0, DAYS - 7, n_drop)
    act_len[drop_ids] = rng.integers(7, 36, n_drop)

    # половина дропперов — новые «одноразовые» карты
    fresh = is_dropper & (rng.random(n) < 0.5)
    account_age[fresh] = rng.integers(3, 60, fresh.sum())
    # у легитимных тоже бывают новые счета
    newbie = (~is_dropper) & (rng.random(n) < 0.06)
    account_age[newbie] = rng.integers(3, 60, newbie.sum())

    # смена номера телефона: день относительно START (отрицательный — до наблюдения)
    phone_change_day = np.full(n, -9999)
    legit_change = (~is_dropper) & (rng.random(n) < 0.04)
    phone_change_day[legit_change] = rng.integers(-60, DAYS, legit_change.sum())
    drop_change = is_dropper & (rng.random(n) < np.where(sophisticated, 0.25, 0.65))
    phone_change_day[drop_change] = act_start[drop_change] - rng.integers(0, 14, drop_change.sum())

    devices = 1 + rng.poisson(0.3, n)
    devices[is_dropper] += rng.poisson(np.where(sophisticated[is_dropper], 0.5, 1.5))

    has_salary = rng.random(n) < np.where(segment == "regular", 0.62, 0.25)
    has_salary[is_dropper] = rng.random(n_drop) < 0.15

    return pd.DataFrame(
        dict(
            client_id=ids,
            age=age,
            segment=segment,
            account_age_at_start=account_age,
            phone_change_day=phone_change_day,
            devices=devices,
            has_salary=has_salary,
            is_dropper=is_dropper,
            confirmed=confirmed,
            sophisticated=sophisticated,
            act_start=act_start,
            act_len=act_len,
        )
    )


def _ts(rng, day, n, night_share=0.04):
    """Время перевода: дневной профиль + доля ночных операций."""
    hour = np.where(
        rng.random(n) < night_share,
        rng.integers(0, 6, n),
        np.clip(rng.normal(15, 3.5, n), 6, 23.99),
    )
    return START + pd.to_timedelta(day, "D") + pd.to_timedelta(hour * 3600, "s")


def legit_transfers(rng, clients: pd.DataFrame) -> pd.DataFrame:
    n = len(clients)
    circle = rng.integers(0, n, (n, 15))          # «круг общения» каждого клиента
    rate = rng.lognormal(np.log(0.18), 0.6, n)    # переводов в день
    k = rng.poisson(rate * DAYS)
    sender = np.repeat(np.arange(n), k)
    receiver = circle[sender, rng.integers(0, 15, len(sender))]
    day = rng.integers(0, DAYS, len(sender))
    big = rng.random(len(sender)) < 0.12                 # аренда, долги, покупка с рук
    amount = np.where(big, rng.lognormal(np.log(20000), 0.9, len(sender)),
                      rng.lognormal(np.log(1500), 0.9, len(sender)))
    # люди любят круглые суммы: ~половина крупных переводов кратна 1000
    amount = np.where((amount > 2000) & (rng.random(len(sender)) < 0.5), np.round(amount, -3),
                      np.round(amount, -1))
    # часть переводов уходит в другие банки (receiver = -1)
    ext = rng.random(len(sender)) < 0.25
    receiver = np.where(ext, -1, receiver)
    df = pd.DataFrame(dict(sender=sender, receiver=receiver, day=day, amount=amount))

    # студенты получают регулярные переводы от «родителей»
    stud = clients.index[clients.segment == "student"].to_numpy()
    reps = DAYS // 14
    s_recv = np.repeat(stud, reps)
    s_send = np.repeat(rng.integers(0, n, len(stud)), reps)
    s_day = np.tile(np.arange(reps) * 14, len(stud)) + rng.integers(0, 3, len(s_recv))
    s_amt = np.round(rng.choice([5000, 10000, 15000, 20000], len(s_recv)), -2)
    parents = pd.DataFrame(dict(sender=s_send, receiver=s_recv, day=s_day, amount=s_amt))

    # продавцы: много входящих от незнакомых, затем вывод в другой банк через 1–5 дней
    sell = clients.index[clients.segment == "seller"].to_numpy()
    k2 = rng.poisson(1.2 * DAYS, len(sell))
    b_recv = np.repeat(sell, k2)
    b_send = rng.integers(0, n, len(b_recv))
    b_day = rng.integers(0, DAYS, len(b_recv))
    b_amt = np.round(rng.lognormal(np.log(2500), 0.7, len(b_recv)), 0)
    buyers = pd.DataFrame(dict(sender=b_send, receiver=b_recv, day=b_day, amount=b_amt))
    payout = (
        buyers.assign(week=buyers.day // 4)
        .groupby(["receiver", "week"], as_index=False)
        .agg(amount=("amount", "sum"))
    )
    payout = pd.DataFrame(
        dict(
            sender=payout.receiver,
            receiver=-1,
            day=np.minimum(payout.week * 4 + rng.integers(1, 6, len(payout)), DAYS - 1),
            amount=np.round(payout.amount * rng.uniform(0.6, 0.95, len(payout)), 0),
        )
    )
    # продажи с рук (Авито и т.п.): ~30% обычных клиентов иногда получают деньги от незнакомых
    # и часто сразу переводят их дальше — ещё один источник «похожих на дропперов» людей
    sellers_casual = clients.index[(clients.segment == "regular") & (rng.random(n) < 0.30)].to_numpy()
    k4 = rng.poisson(0.04 * DAYS, len(sellers_casual))
    a_recv = np.repeat(sellers_casual, k4)
    a_day = rng.integers(0, DAYS, len(a_recv))
    a_amt = np.round(rng.lognormal(np.log(6000), 1.0, len(a_recv)), -2)
    avito = pd.DataFrame(dict(sender=rng.integers(0, n, len(a_recv)), receiver=a_recv, day=a_day, amount=a_amt))
    resend = rng.random(len(a_recv)) < 0.5
    avito_out = pd.DataFrame(dict(sender=a_recv[resend], receiver=-1, day=a_day[resend],
                                  amount=np.round(a_amt[resend] * rng.uniform(0.7, 1.0, resend.sum()), 0)))

    # P2P-трейдеры криптовалюты: легитимны, но поведение почти как у дропперов —
    # много незнакомых отправителей, круглые суммы, быстрый вывод во внешний банк.
    tr = clients.index[clients.segment == "p2p_trader"].to_numpy()
    k3 = rng.poisson(3.0 * DAYS, len(tr))
    t_recv = np.repeat(tr, k3)
    t_send = rng.integers(0, n, len(t_recv))
    t_day = rng.integers(0, DAYS, len(t_recv))
    t_amt = np.round(rng.lognormal(np.log(12000), 0.9, len(t_recv)), -3)
    trades = pd.DataFrame(dict(sender=t_send, receiver=t_recv, day=t_day, amount=t_amt))
    t_out = pd.DataFrame(dict(sender=t_recv, receiver=-1,
                              day=np.minimum(t_day + rng.integers(0, 2, len(t_recv)), DAYS - 1),
                              amount=np.round(t_amt * rng.uniform(0.9, 1.0, len(t_recv)), 0)))
    out = pd.concat([df, parents, buyers, payout, trades, t_out, avito, avito_out], ignore_index=True)
    out["ts"] = _ts(rng, out.day.to_numpy(), len(out))
    out["is_fraud"] = False
    return out


def dropper_flows(rng, clients: pd.DataFrame):
    """Входящие от жертв и быстрый вывод: дальше по цепочке, наличными или во внешний банк."""
    n = len(clients)
    drops = clients[clients.is_dropper]
    victims_pool = clients.index[clients.age > 45].to_numpy()
    rows, cash = [], []
    for c in drops.itertuples():
        soph = c.sophisticated
        days = np.arange(c.act_start, min(c.act_start + c.act_len, DAYS))
        k = rng.poisson(1.0 if soph else 2.2, len(days))
        n_in = k.sum()
        if n_in == 0:
            continue
        in_day = np.repeat(days, k)
        senders = np.where(
            rng.random(n_in) < 0.5,
            rng.choice(victims_pool, n_in),       # жертвы чаще старшего возраста
            rng.integers(0, n, n_in),
        )
        amt = rng.lognormal(np.log(9000 if soph else 15000), 0.9, n_in)
        amt = np.where(rng.random(n_in) < (0.4 if soph else 0.65), np.round(amt, -3), np.round(amt, -1))
        in_ts = _ts(rng, in_day, n_in, night_share=0.12)
        rows.append(pd.DataFrame(dict(sender=senders, receiver=c.client_id, day=in_day,
                                      amount=amt, ts=in_ts, is_fraud=True)))

        # вывод: доля 85–99%, задержка — минуты у простых, часы/дни у продвинутых
        delay_min = rng.lognormal(np.log(1800 if soph else 25), 1.0, n_in)
        out_ts = in_ts + pd.to_timedelta(delay_min, "m")
        out_amt = np.round(amt * (rng.uniform(0.5, 0.9, n_in) if soph else rng.uniform(0.85, 0.99, n_in)), 0)
        how = rng.choice(["chain", "external", "cash"], n_in, p=[0.25, 0.45, 0.30])
        drop_ids = drops.client_id.to_numpy()
        chain_recv = rng.choice(drop_ids, n_in)
        recv = np.where(how == "chain", chain_recv, -1)
        m = how != "cash"
        out_day = np.minimum((out_ts - START).days, DAYS - 1)
        rows.append(pd.DataFrame(dict(sender=c.client_id, receiver=recv[m], day=out_day[m],
                                      amount=out_amt[m], ts=out_ts[m], is_fraud=True)))
        cash.append(pd.DataFrame(dict(client_id=c.client_id, ts=out_ts[~m], amount=out_amt[~m])))
    return pd.concat(rows, ignore_index=True), pd.concat(cash, ignore_index=True)


def legit_cash(rng, clients: pd.DataFrame) -> pd.DataFrame:
    n = len(clients)
    k = rng.poisson(0.05 * DAYS, n)
    cid = np.repeat(np.arange(n), k)
    day = rng.integers(0, DAYS, len(cid))
    return pd.DataFrame(dict(client_id=cid, ts=_ts(rng, day, len(cid)),
                             amount=np.round(rng.lognormal(np.log(5000), 0.6, len(cid)), -2)))


def main(n_clients: int, seed: int, out: Path) -> None:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    clients = make_clients(rng, n_clients)
    legit = legit_transfers(rng, clients)
    fraud, dcash = dropper_flows(rng, clients)
    transfers = pd.concat([legit, fraud], ignore_index=True)
    transfers = transfers[transfers.sender != transfers.receiver]
    transfers = transfers[transfers.ts < START + pd.Timedelta(days=DAYS)]
    transfers = transfers.sort_values("ts").reset_index(drop=True)
    transfers.insert(0, "transfer_id", np.arange(len(transfers)))
    cash = pd.concat([legit_cash(rng, clients), dcash], ignore_index=True)
    cash = cash[cash.ts < START + pd.Timedelta(days=DAYS)].sort_values("ts")

    clients.to_parquet(out / "clients.parquet", index=False)
    transfers.drop(columns="day").to_parquet(out / "transfers.parquet", index=False)
    cash.to_parquet(out / "cash.parquet", index=False)
    print(f"clients={len(clients):,}  droppers={clients.is_dropper.sum():,}  "
          f"transfers={len(transfers):,}  fraud_transfers={transfers.is_fraud.sum():,}  "
          f"cash_ops={len(cash):,}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clients", type=int, default=40_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    a = p.parse_args()
    main(a.clients, a.seed, a.out)
