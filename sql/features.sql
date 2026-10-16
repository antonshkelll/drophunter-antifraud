-- Витрина признаков клиента за период [$p_start, $p_end).
-- Таблицы: clients, transfers(transfer_id, sender, receiver, ts, amount, is_fraud), cash(client_id, ts, amount)
-- Одна строка = один клиент в одном периоде. Метка: был ли клиент активным дроппером в периоде.

WITH
inc AS (          -- входящие переводы
    SELECT receiver AS client_id, sender, ts, amount
    FROM transfers
    WHERE receiver >= 0 AND ts >= $p_start AND ts < $p_end
),
outg AS (         -- исходящие: переводы + снятие наличных
    SELECT sender AS client_id, ts, amount, 'transfer' AS kind, (receiver = -1) AS external
    FROM transfers WHERE ts >= $p_start AND ts < $p_end
    UNION ALL
    SELECT client_id, ts, amount, 'cash', FALSE
    FROM cash WHERE ts >= $p_start AND ts < $p_end
),
-- отправители, которых клиент «знал» до периода (хоть раз переводили ему раньше)
known AS (
    SELECT DISTINCT receiver AS client_id, sender
    FROM transfers WHERE receiver >= 0 AND ts < $p_start
),
inc_feat AS (
    SELECT
        i.client_id,
        count(*)                                              AS in_cnt,
        sum(i.amount)                                         AS in_sum,
        count(DISTINCT i.sender)                              AS in_unique_senders,
        avg(CASE WHEN i.amount % 1000 = 0 THEN 1 ELSE 0 END)  AS in_round_share,
        avg(CASE WHEN hour(i.ts) < 6 THEN 1 ELSE 0 END)       AS in_night_share,
        avg(CASE WHEN k.sender IS NULL THEN 1 ELSE 0 END)     AS in_new_sender_share,
        avg(CASE WHEN c.age >= 55 THEN 1 ELSE 0 END)          AS in_elderly_sender_share,
        max(i.amount)                                         AS in_max_amount
    FROM inc i
    LEFT JOIN known k   ON k.client_id = i.client_id AND k.sender = i.sender
    LEFT JOIN clients c ON c.client_id = i.sender
    GROUP BY i.client_id
),
daily AS (        -- пиковая «воронка» за один день: сколько разных отправителей
    SELECT client_id, max(n) AS max_daily_senders
    FROM (SELECT client_id, date_trunc('day', ts) d, count(DISTINCT sender) n FROM inc GROUP BY 1, 2)
    GROUP BY 1
),
out_feat AS (
    SELECT
        client_id,
        count(*)                                                       AS out_cnt,
        sum(amount)                                                    AS out_sum,
        sum(CASE WHEN kind = 'cash' THEN amount ELSE 0 END)            AS cash_sum,
        sum(CASE WHEN external THEN amount ELSE 0 END)                 AS external_sum,
        avg(CASE WHEN hour(ts) < 6 THEN 1 ELSE 0 END)                  AS out_night_share
    FROM outg GROUP BY 1
),
-- сколько минут проходит от входящего перевода до ближайшего следующего списания
lag_to_out AS (
    SELECT i.client_id,
           median(date_diff('minute', i.ts, o.ts))                       AS median_min_to_out,
           avg(CASE WHEN date_diff('minute', i.ts, o.ts) <= 60 THEN 1 ELSE 0 END) AS out_within_1h_share
    FROM inc i
    ASOF JOIN outg o ON i.client_id = o.client_id AND o.ts >= i.ts
    GROUP BY 1
),
label AS (
    SELECT client_id,
           is_dropper AND confirmed
             AND act_start < date_diff('day', TIMESTAMP '2026-04-01', $p_end::TIMESTAMP)
             AND act_start + act_len > date_diff('day', TIMESTAMP '2026-04-01', $p_start::TIMESTAMP)
             AS target
    FROM clients
)
SELECT
    c.client_id,
    c.age,
    c.account_age_at_start + date_diff('day', TIMESTAMP '2026-04-01', $p_start::TIMESTAMP) AS account_age_days,
    -- дни с последней смены номера к концу периода (смена в будущем не видна → 9999)
    CASE WHEN c.phone_change_day > -9999
          AND c.phone_change_day < date_diff('day', TIMESTAMP '2026-04-01', $p_end::TIMESTAMP)
         THEN date_diff('day', TIMESTAMP '2026-04-01', $p_end::TIMESTAMP) - c.phone_change_day
         ELSE 9999 END                                             AS days_since_phone_change,
    c.devices,
    c.has_salary::INT                                              AS has_salary,
    coalesce(i.in_cnt, 0)                                          AS in_cnt,
    coalesce(i.in_sum, 0)                                          AS in_sum,
    coalesce(i.in_unique_senders, 0)                               AS in_unique_senders,
    coalesce(i.in_round_share, 0)                                  AS in_round_share,
    coalesce(i.in_night_share, 0)                                  AS in_night_share,
    coalesce(i.in_new_sender_share, 0)                             AS in_new_sender_share,
    coalesce(i.in_elderly_sender_share, 0)                         AS in_elderly_sender_share,
    coalesce(i.in_max_amount, 0)                                   AS in_max_amount,
    coalesce(d.max_daily_senders, 0)                               AS max_daily_senders,
    coalesce(o.out_cnt, 0)                                         AS out_cnt,
    coalesce(o.out_sum, 0)                                         AS out_sum,
    coalesce(o.cash_sum / nullif(o.out_sum, 0), 0)                 AS cash_share,
    coalesce(o.external_sum / nullif(o.out_sum, 0), 0)             AS external_share,
    coalesce(o.out_night_share, 0)                                 AS out_night_share,
    coalesce(o.out_sum / nullif(i.in_sum, 0), 0)                   AS passthrough_ratio,
    coalesce(l.median_min_to_out, 99999)                           AS median_min_to_out,
    coalesce(l.out_within_1h_share, 0)                             AS out_within_1h_share,
    lb.target::INT                                                 AS target
FROM clients c
LEFT JOIN inc_feat i   USING (client_id)
LEFT JOIN daily d      USING (client_id)
LEFT JOIN out_feat o   USING (client_id)
LEFT JOIN lag_to_out l USING (client_id)
JOIN label lb          USING (client_id)
WHERE coalesce(i.in_cnt, 0) > 0          -- скорим только клиентов с входящими в периоде
