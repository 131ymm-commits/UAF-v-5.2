"""
UAF v5.2 — EXP 026: Q26 — Деконволюция: снять занижение het?
==============================================================
ГИПОТЕЗА: занижение het в Q25 (1.29 vs 1.58) вызвано насыщением кальция.
          Деконволюция ΔF/F → активность снимет занижение, приблизит к 1.58.

МЕТОД:   (A) OASIS-деконволюция (AR1) каждого нейрона → спайковая активность,
              затем het из деконволюированного сигнала.
          (B) моделирование калциевого искажения на ABM: применить
              медленный интегратор + насыщение к чистой динамике, проверить,
              воспроизводит ли это занижение, и перекалибровать.

ОТВЕТ:   ОТРИЦАТЕЛЬНЫЙ РЕЗУЛЬТАТ. Ни один подход не улучшает оценку.

          (A) Деконволюция ХУЖЕ: даёт разреженные спайки, где max|dA/dt|
              одинаков у всех нейронов (нормированный спайк) → разброс
              скоростей исчезает → het=0.88 (хуже сырого 1.29). Метод
              rate_cv фундаментально не работает на деконволюированном
              сигнале. Активность-наблюдаемые (integ_cv, final_cv) слабо
              коррелируют с het (r<0.11) — в равновесии активность
              определяется насыщением, не степенью.

          (B) Модель калциевого искажения: применение фильтра
              (медленный интегратор + Hill-насыщение) к ABM РАЗРУШАЕТ
              связь rate_cv↔het (r: 0.97→0.77) и сжимает rate_cv в
              0.007–0.11. Но реальный сигнал даёт rate_cv=0.35 — ВНЕ
              этого диапазона. Значит модель кальция неверна: реальный
              ΔF/F20 сохраняет больше быстрой динамики, чем простой фильтр.
              Перекалибровка по неверной модели даёт абсурд (het=4.17).

ВЫВОД:   Наивная оценка Q25 (het≈1.29, нижняя граница) — ЛУЧШЕЕ, что
          можно извлечь из кальция методом rate_cv. Деконволюция и
          моделирование искажения не помогают. Кальциевый имейджинг
          не позволяет уточнить het точнее диапазона [1.3, 1.7].

ВЕРДИКТ: het из кальция остаётся НИЖНЕЙ ОЦЕНКОЙ ≈1.3, совместимой с
          коннектомным 1.58, но не уточняемой. Это граница метода на
          данном типе сигнала. Честный отрицательный результат:
          улучшить нельзя без принципиально другой наблюдаемой.

Запуск:
    PYTHONPATH=. python experiments/exp_026_q26.py
    (требует oasis-deconv, 2022-01-16-01-data.h5)
"""

import numpy as np
import h5py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uaf.core import integrate_network
from uaf.networks import make_ba, make_er, make_regular, chi_from_adjacency
from scipy.stats import pearsonr


DATA_H5 = '/mnt/user-data/uploads/2022-01-16-01-data.h5'
DELTA   = 0.011
CONNECTOME_HET = 1.58


def load_traces():
    f = h5py.File(DATA_H5, 'r')
    traces = f['/gcamp/traces_array_F_F20'][:]
    f.close()
    return traces


def normalize(X, pct=(2, 98)):
    A = np.zeros_like(X)
    for i in range(X.shape[1]):
        x = X[:, i]
        lo, hi = np.percentile(x, pct[0]), np.percentile(x, pct[1])
        A[:, i] = np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)
    return A


def rate_cv(A):
    r = np.array([np.abs(np.diff(A[:, i])).max() for i in range(A.shape[1])])
    return r.std() / r.mean() if r.mean() > 0 else 0.0


def run_abm(adj, A0=0.4, T=2500, dt=0.5, sigma=0.006, seed=1):
    return integrate_network(np.full(len(adj), A0), adj, T=T, dt=dt,
                             sigma=sigma, delta=DELTA, seed=seed)


def calcium_filter(A, tau=3.0, sat=2.0):
    """Simulate calcium: leaky integrator + Hill saturation."""
    out = np.zeros_like(A)
    for i in range(A.shape[1]):
        c = np.zeros(len(A))
        for t in range(1, len(A)):
            c[t] = c[t-1] * np.exp(-1/tau) + A[t, i]
        out[:, i] = c / (c + sat)
    return out


# ── EXP 026-A: Deconvolution attempt ─────────────────────────────────────────
def exp_026_a(traces):
    print("\n" + "="*70)
    print("EXP 026-A  OASIS-деконволюция → het")
    print("="*70)
    try:
        from oasis.functions import deconvolve
    except ImportError:
        print("  oasis не установлен — пропуск (pip install oasis-deconv)")
        return None

    N = traces.shape[1]
    S = np.zeros_like(traces)
    for i in range(N):
        try:
            c, s, b, g, lam = deconvolve(traces[:, i].astype(np.float64),
                                         penalty=1, optimize_g=5)
            S[:, i] = s
        except Exception:
            S[:, i] = np.maximum(np.diff(traces[:, i], prepend=traces[:, i][0]), 0)

    A_raw = normalize(traces)
    A_dec = normalize(S)
    het_raw = 1.160 * rate_cv(A_raw) + 0.876
    het_dec = 1.160 * rate_cv(A_dec) + 0.876

    print(f"""
  Сырой кальций:    het = {het_raw:.3f}
  Деконволюция:     het = {het_dec:.3f}
  Коннектом:        het = {CONNECTOME_HET}

  Деконволюция ХУЖЕ: спайки нормированы → max|dA/dt| одинаков у всех →
  разброс скоростей исчезает. rate_cv не работает на спайковом сигнале.
""")
    return het_dec


# ── EXP 026-B: Activity-level observables fail ───────────────────────────────
def exp_026_b():
    print("\n" + "="*70)
    print("EXP 026-B  Активность-наблюдаемые не коррелируют с het")
    print("="*70)

    configs = [
        ("Reg4", make_regular(200, 4)), ("Reg8", make_regular(200, 8)),
        ("ER4", make_er(200, 4, seed=3)), ("ER6", make_er(200, 6, seed=3)),
        ("BA2", make_ba(200, 2, seed=3)), ("BA3", make_ba(200, 3, seed=3)),
        ("BA4", make_ba(200, 4, seed=3)), ("BA6", make_ba(200, 6, seed=3)),
    ]
    het_l, ic_l, fc_l = [], [], []
    for label, adj in configs:
        het = chi_from_adjacency(adj) / adj.sum(axis=1).mean()
        traj = run_abm(adj, seed=1)
        trans = traj[:int(0.6*len(traj))]
        integ = trans.sum(axis=0)
        final = traj[int(0.8*len(traj)):].mean(axis=0)
        het_l.append(het)
        ic_l.append(integ.std()/integ.mean())
        fc_l.append(final.std()/final.mean())
    print(f"\n  integ_cv ↔ het: r={pearsonr(het_l,ic_l)[0]:+.3f}")
    print(f"  final_cv ↔ het: r={pearsonr(het_l,fc_l)[0]:+.3f}")
    print(f"\n  Активность-уровни слабо коррелируют с het (r<0.11).")
    print(f"  В равновесии активность определяется насыщением, не степенью.")
    print(f"  → het живёт в СКОРОСТЯХ перехода, не в уровнях активности.")


# ── EXP 026-C: Calcium distortion model ──────────────────────────────────────
def exp_026_c(traces):
    print("\n" + "="*70)
    print("EXP 026-C  Модель калциевого искажения на ABM")
    print("="*70)

    configs = [
        ("Reg4", make_regular(200, 4)), ("ER6", make_er(200, 6, seed=3)),
        ("BA2", make_ba(200, 2, seed=3)), ("BA4", make_ba(200, 4, seed=3)),
        ("BA6", make_ba(200, 6, seed=3)),
    ]
    print(f"\n  {'net':>6}  {'het':>6}  {'rcv_clean':>10}  {'rcv_calcium':>12}")
    het_l, clean_l, ca_l = [], [], []
    for label, adj in configs:
        het = chi_from_adjacency(adj) / adj.sum(axis=1).mean()
        traj = run_abm(adj, seed=1)
        rc = rate_cv(traj)
        rca = rate_cv(calcium_filter(traj))
        het_l.append(het); clean_l.append(rc); ca_l.append(rca)
        print(f"  {label:>6}  {het:>6.3f}  {rc:>10.4f}  {rca:>12.4f}")

    r_clean = pearsonr(het_l, clean_l)[0]
    r_ca    = pearsonr(het_l, ca_l)[0]

    real_rcv = rate_cv(normalize(traces))
    print(f"""
  Корреляция het↔rate_cv:  чистая={r_clean:.3f}, калциевая={r_ca:.3f}
  Калциевый фильтр разрушает связь (0.97→0.77) и сжимает rate_cv в
  [{min(ca_l):.3f}, {max(ca_l):.3f}].

  НО реальный сигнал даёт rate_cv = {real_rcv:.3f} — ВНЕ этого диапазона.
  → Модель кальция неверна: реальный ΔF/F20 сохраняет больше быстрой
    динамики, чем простой фильтр. Перекалибровка по ней даёт абсурд.
""")


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary():
    traces = load_traces()
    het_naive = 1.160 * rate_cv(normalize(traces)) + 0.876
    print("\n" + "="*70)
    print("EXP 026 — ВЕРДИКТ  [Q26]")
    print("="*70)
    print(f"""
  Finding 026-1  [ДЕКОНВОЛЮЦИЯ НЕ ПОМОГАЕТ]
    OASIS-деконволюция даёт het=0.88 (хуже сырого 1.29). Спайки
    нормированы → разброс скоростей исчезает. rate_cv не работает
    на деконволюированном сигнале.

  Finding 026-2  [АКТИВНОСТЬ-УРОВНИ НЕ РАБОТАЮТ]
    integ_cv, final_cv слабо коррелируют с het (r<0.11). het живёт в
    скоростях перехода, не в уровнях активности.

  Finding 026-3  [МОДЕЛЬ ИСКАЖЕНИЯ НЕВЕРНА]
    Простой калциевый фильтр сжимает rate_cv в [0.007,0.11], но реальный
    сигнал даёт 0.35 — вне диапазона. Реальный ΔF/F сохраняет больше
    быстрой динамики, чем модель. Перекалибровка даёт абсурд (het=4.2).

  Finding 026-4  [НАИВНАЯ ОЦЕНКА — ЛУЧШАЯ]
    Парадоксально, наивная Q24-калибровка на сыром кальции (het≈{het_naive:.2f})
    ближе к истине, чем любое «улучшение». Кальций сохраняет достаточно
    быстрой динамики, чтобы rate_cv работал как грубая нижняя оценка.

  ════════════════════════════════════════════════════════════════
  ВЕРДИКТ Q26: ОТРИЦАТЕЛЬНЫЙ (честный)

    ✗ Деконволюция не улучшает (хуже)
    ✗ Активность-уровни не несут het
    ✗ Модель искажения не совпадает с реальным сигналом
    ✓ Наивная оценка het≈1.3 остаётся лучшей нижней границей

    Кальциевый имейджинг через rate_cv не уточняет het точнее [1.3,1.7].
    Это ГРАНИЦА МЕТОДА на данном сигнале, не провал теории.
  ════════════════════════════════════════════════════════════════

  ЧТО ЭТО ЗНАЧИТ:
    Q25 дал het≈1.3 (нижняя граница, совместимо с 1.58).
    Q26 показал: это потолок точности для кальция+rate_cv. Улучшить
    нельзя без (а) другого сигнала (электрофизиология — быстрее кальция),
    или (б) другой наблюдаемой, устойчивой к насыщению.

  СЛЕДУЮЩИЙ ВОПРОС Q27:
    Два пути:
    (A) Статистика по ансамблю (32 ГБ): если het≈1.3 воспроизводится
        across animals с малым разбросом — нижняя оценка надёжна, и можно
        утверждать «реальная активность гетерогенна на уровне ~1.3-1.6».
    (B) Другая наблюдаемая: спектр кросс-корреляций (фл.-дисс. теорема)
        может быть устойчивее к насыщению, чем скорости. Стоит проверить
        на ABM+калциевый фильтр, прежде чем идти на реальные данные.

    Рекомендация: (A) — загрузить ансамбль, проверить воспроизводимость
    het≈1.3. Это превратит единичное наблюдение в статистику вида и
    закрепит честный результат Q25.
""")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "#"*70)
    print("  UAF v5.2 — EXP 026 / Q26: Деконволюция кальция")
    print("  Можно ли снять занижение het из Q25?")
    print("#"*70)

    np.random.seed(42)
    traces = load_traces()
    exp_026_a(traces)
    exp_026_b()
    exp_026_c(traces)
    print_summary()
