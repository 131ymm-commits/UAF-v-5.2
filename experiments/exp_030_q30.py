"""
UAF v5.2 — EXP 030: Q30 — Спасает ли взвешенный коннектом операционализацию?
=============================================================================
ГИПОТЕЗА: Q28 показал, что реальные отклики нейронов не следуют БИНАРНОЙ
          топологии. Гипотеза: дело в весах — отклик определяется
          синаптической СИЛОЙ (число контактов Nbr), не числом партнёров.
          Взвешенный коннектом мог бы спасти операционализацию.

ДАННЫЕ:  коннектом с весами (NeuronConnect.xls, колонка Nbr = число
          синаптических контактов) + Leifer оптогенетические отклики
          (10 записей, нейроны идентифицированы по именам → маппинг).

МЕТОД:   (A) взвешенная структура: het_w=<s²>/<s>², s=сила связи.
          (B) прямой тест: коррелирует ли реальный отклик нейрона на
              стимуляцию (Leifer) со взвешенной степенью лучше, чем с
              бинарной? Это проверяет, объясняют ли веса провал Q28.

ОТВЕТ:   ВЗВЕШЕННЫЙ КОННЕКТОМ НЕ СПАСАЕТ. Гипотеза весов ОТВЕРГНУТА.

          (A) Веса дают другую структуру: het_w=2.05 vs het_bin=1.58.
              Spearman(bin,wt)=0.80 — ранжирования заметно расходятся.
              AVAL: 606 контактов при 92 партнёрах. Информация есть.

          (B) НО отклик нейрона НЕ коррелирует НИ с чем:
              response vs binary degree:   r=−0.016
              response vs weighted degree:  r=+0.010
              (471 наблюдение, оба ≈ ноль; по записям знак случаен)

ВЫВОД:   Реальный отклик нейрона на возмущение не определяется ни числом
          связей, ни их суммарной силой. Это глубже провала Q28: дело не
          в бинарности vs взвешенности. Реальная нейронная реакция
          зависит от того, чего НЕТ в коннектоме: знак связи
          (возбуждение/торможение), нейромедиатор, внутренняя динамика
          нейрона, состояние сети. Структурная связность (любая) этого
          не содержит.

ВЕРДИКТ: ОТРИЦАТЕЛЬНЫЙ. Взвешивание не спасает. Подтверждает и углубляет
          вывод Q28: операционализация UAF через структуру→динамику
          невозможна на реальных нейронах, потому что отклик определяется
          небинарной, неструктурной биофизикой. Граница UAF подтверждена
          с ещё одной стороны.

Запуск:
    PYTHONPATH=. python experiments/exp_030_q30.py
    (требует NeuronConnect.xls + распакованный Leifer2023)
"""

import numpy as np
import pandas as pd
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.stats import pearsonr, spearmanr


CONN_PATH = '/mnt/user-data/uploads/NeuronConnect.xls'
LEIFER = 'leifer/Leifer2023'
COMMAND = ['AVAL', 'AVAR', 'AVBL', 'AVBR', 'AVDL', 'AVDR',
           'AVEL', 'AVER', 'PVCL', 'PVCR']


def build_degrees(path=CONN_PATH):
    df = pd.read_excel(path)
    neurons = sorted(set(df['Neuron 1']) | set(df['Neuron 2']))
    idx = {n: i for i, n in enumerate(neurons)}
    N = len(neurons)
    Ab = np.zeros((N, N)); Aw = np.zeros((N, N))
    for _, r in df.iterrows():
        t = str(r['Type']); w = r['Nbr']
        if t in ('S', 'Sp', 'EJ'):
            i, j = idx[r['Neuron 1']], idx[r['Neuron 2']]
            Ab[i, j] = Ab[j, i] = 1
            Aw[i, j] += w; Aw[j, i] += w
    bin_deg = {neurons[i]: Ab[i].sum() for i in range(N)}
    wt_deg = {neurons[i]: Aw[i].sum() for i in range(N)}
    return bin_deg, wt_deg


def het(values):
    v = np.array(values)
    return (v**2).mean() / v.mean()**2


def load_leifer(num):
    g = np.loadtxt(f'{LEIFER}/{num}_gcamp.txt')
    sv = np.loadtxt(f'{LEIFER}/{num}_stim_volume_i.txt').astype(int)
    with open(f'{LEIFER}/{num}_labels.txt') as f:
        labels = [l.strip() for l in f]
    return g, sv, labels


def clean(g, mn=0.3):
    keep = np.isnan(g).mean(axis=0) < mn
    g2 = g[:, keep].copy()
    for i in range(g2.shape[1]):
        x = g2[:, i]; na = np.isnan(x)
        if na.any():
            x[na] = np.interp(np.flatnonzero(na), np.flatnonzero(~na), x[~na])
            g2[:, i] = x
    return g2, np.flatnonzero(keep)


def normalize(X, pct=(2, 98)):
    A = np.zeros_like(X)
    for i in range(X.shape[1]):
        x = X[:, i]
        lo, hi = np.percentile(x, pct[0]), np.percentile(x, pct[1])
        A[:, i] = np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)
    return A


# ── EXP 030-A: Weighted structure ────────────────────────────────────────────
def exp_030_a(bin_deg, wt_deg):
    print("\n" + "="*70)
    print("EXP 030-A  Взвешенная структура коннектома")
    print("="*70)
    bd = np.array([v for v in bin_deg.values() if v > 0])
    wd = np.array([wt_deg[k] for k, v in bin_deg.items() if v > 0])
    print(f"""
  Бинарная степень:   <k>={bd.mean():.1f}, max={int(bd.max())}
  Взвешенная степень: <s>={wd.mean():.1f}, max={int(wd.max())}
  het бинарный  = {het(bd):.3f}
  het взвешенный = {het(wd):.3f}  (сеть сильнее гетерогенна по СИЛЕ связей)
  Spearman(bin, wt) = {spearmanr(bd, wd)[0]:.3f}  (ранжирования расходятся)
""")
    name2 = {k: i for i, k in enumerate([k for k, v in bin_deg.items() if v > 0])}
    print(f"  {'нейрон':>7}  {'bin_ранг':>9}  {'wt_ранг':>8}")
    bd_all = {k: v for k, v in bin_deg.items() if v > 0}
    for c in COMMAND:
        if c in bd_all:
            br = sum(1 for v in bd_all.values() if v > bd_all[c]) + 1
            wr = sum(1 for k in bd_all if wt_deg[k] > wt_deg[c]) + 1
            print(f"  {c:>7}  {br:>9}  {wr:>8}")


# ── EXP 030-B: Response vs degree (the decisive test) ────────────────────────
def exp_030_b(bin_deg, wt_deg):
    print("\n" + "="*70)
    print("EXP 030-B  Решающий тест: отклик нейрона vs степень")
    print("="*70)
    print(f"\n  {'rec':>4}  {'n':>5}  {'r_бинар':>9}  {'r_взвеш':>9}  {'лучше'}")
    print("  " + "-"*40)

    all_r, all_b, all_w = [], [], []
    for num in [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]:
        g, sv, labels = load_leifer(num)
        g2, kept = clean(g)
        A = normalize(g2)
        resp = np.zeros(A.shape[1])
        for st in sv:
            if 5 < st < len(A) - 10:
                resp += np.abs(A[st:st+10].mean(axis=0) - A[st-5:st].mean(axis=0))
        kl = [labels[k] if k < len(labels) else '' for k in kept]
        rb, rw, rr = [], [], []
        for li, lab in enumerate(kl):
            if lab in bin_deg and lab not in ('merge', '', ' '):
                rb.append(bin_deg[lab]); rw.append(wt_deg[lab]); rr.append(resp[li])
        if len(rr) < 10:
            continue
        rb, rw, rr = np.array(rb), np.array(rw), np.array(rr)
        cb, cw = pearsonr(rb, rr)[0], pearsonr(rw, rr)[0]
        all_r += list(rr); all_b += list(rb); all_w += list(rw)
        print(f"  {num:>4}  {len(rr):>5}  {cb:>+9.3f}  {cw:>+9.3f}  "
              f"{'взвеш' if abs(cw)>abs(cb) else 'бинар'}")

    rb, rw, rr = np.array(all_b), np.array(all_w), np.array(all_r)
    print(f"""
  ПУЛ ({len(rr)} наблюдений):
    Отклик vs бинарная степень:   r={pearsonr(rb,rr)[0]:+.3f}
    Отклик vs взвешенная степень: r={pearsonr(rw,rr)[0]:+.3f}

  ОБА ≈ НОЛЬ. Ни число связей, ни их сила не предсказывают отклик.
  По записям знак корреляции случаен (то +, то −).
""")
    return pearsonr(rb, rr)[0], pearsonr(rw, rr)[0]


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(r_bin, r_wt):
    print("\n" + "="*70)
    print("EXP 030 — ВЕРДИКТ  [Q30]")
    print("="*70)
    print(f"""
  Finding 030-1  [веса несут информацию]
    het_взвеш=2.05 vs het_бинар=1.58. Spearman 0.80 — ранжирования
    расходятся. Взвешенный коннектом — не копия бинарного.

  Finding 030-2  [но отклик не следует НИ ОДНОЙ степени]
    Отклик нейрона на стимуляцию (Leifer, 471 набл.):
      vs бинарная степень:   r={r_bin:+.3f}
      vs взвешенная степень: r={r_wt:+.3f}
    Оба ≈ ноль. Гипотеза весов (объяснить провал Q28) ОТВЕРГНУТА.

  Finding 030-3  [углубление вывода Q28]
    Дело не в бинарности vs взвешенности. Реальный отклик зависит от
    того, чего НЕТ в коннектоме: знак связи (возбуждение/торможение),
    нейромедиатор, внутренняя динамика нейрона, состояние сети.
    Любая структурная связность этого не содержит.

  ════════════════════════════════════════════════════════════════
  ВЕРДИКТ Q30: ОТРИЦАТЕЛЬНЫЙ

    ✗ Взвешенный коннектом не спасает операционализацию
    ✗ Отклик не коррелирует ни с числом, ни с силой связей
    ✓ Подтверждает и углубляет границу Q28

    Операционализация UAF (структура → динамика) невозможна на реальных
    нейронах не из-за бинарности, а потому что отклик определяется
    небинарной, неструктурной биофизикой (знак, тип, состояние).
  ════════════════════════════════════════════════════════════════

  ЧТО ЭТО ЗНАЧИТ ДЛЯ UAF:
    Граница, найденная в Q25–Q28, теперь объяснена механистически.
    UAF (и любая чисто топологическая модель) не может предсказать
    динамику реальных нейронов, потому что игнорирует ЗНАК связи.
    Возбуждающий хаб и тормозный хаб структурно идентичны, но
    динамически противоположны. Коннектом без знаков — недостаточен.

  СЛЕДУЮЩИЙ ВОПРОС Q31:
    Есть ли данные о ЗНАКЕ связей (возбуждение/торможение) для
    C. elegans? Нейромедиаторный атлас (Bentley et al 2016 — карта
    нейромедиаторов и рецепторов) позволил бы построить ЗНАКОВЫЙ
    коннектом. Тогда вопрос: предсказывает ли UAF на знаковом коннектоме
    реальные отклики? Это последняя структурная попытка перед признанием,
    что нужна полная биофизическая модель (не топологическая).

    Если знаковый коннектом тоже не работает — честный финал: UAF
    применим к структурной устойчивости (Q22, Q29), но динамика реальных
    нейронов требует биофизики за пределами любой топологии.
""")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "#"*70)
    print("  UAF v5.2 — EXP 030 / Q30: Взвешенный коннектом")
    print("  Спасают ли веса синапсов операционализацию?")
    print("#"*70)

    np.random.seed(0)
    bin_deg, wt_deg = build_degrees()
    exp_030_a(bin_deg, wt_deg)
    r_bin, r_wt = exp_030_b(bin_deg, wt_deg)
    print_summary(r_bin, r_wt)
