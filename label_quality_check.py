"""
Compare label quality (fakeout rate, label count) across different k_atr values.
Higher k = stricter breakout confirmation = fewer but cleaner labels.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import yaml
import copy
import pandas as pd
import data as data_mod
import labels as label_mod

cfg = yaml.safe_load(open("config.yaml"))

# Load all years once
print("Loading data...")
dfs = {}
for yr in [2022, 2023, 2024, 2025]:
    try:
        dfs[yr] = data_mod.load_csv(f"Data/BTCUSDT/full_year/{yr}_1m.csv")
        print(f"  {yr}: {len(dfs[yr]):,} bars")
    except FileNotFoundError:
        print(f"  {yr}: not found, skipping")

k_values = [5, 7, 8, 10, 12, 15]

print(f"\n{'='*90}")
print(f"  k_atr comparison  (H_horizon={cfg['labels']['H_horizon']}, L_range={cfg['labels']['L_range']})")
print(f"  UP/DOWN% = fraction of bars labeled as breakout  |  Fake% = fakeouts / (fakeouts+labels)")
print(f"{'='*90}")
print(f"  {'k':>4}  {'Year':>5}  {'Total':>8}  {'UP':>7}  {'DOWN':>7}  {'Break%':>7}  {'FakeUP%':>8}  {'FakeDN%':>8}")
print(f"  {'-'*74}")

for k in k_values:
    cfg_k = copy.deepcopy(cfg)
    cfg_k["labels"]["k_atr"] = k

    combined_rows = []
    for yr, raw in dfs.items():
        lab = label_mod.compute_labels(raw, cfg_k)
        s   = lab["label"].dropna()
        n   = len(s)
        up  = int((s == 2).sum())
        dn  = int((s == 0).sum())
        fk_up = int(lab["up_fakeout"].sum())
        fk_dn = int(lab["down_fakeout"].sum())
        brk_total = up + dn
        fake_up_pct = fk_up / (fk_up + up + 1e-9) * 100
        fake_dn_pct = fk_dn / (fk_dn + dn + 1e-9) * 100
        print(f"  {k:>4}  {yr:>5}  {n:>8,}  {up/n*100:>6.1f}%  {dn/n*100:>6.1f}%  "
              f"{brk_total/n*100:>6.1f}%  {fake_up_pct:>7.1f}%  {fake_dn_pct:>7.1f}%")
        combined_rows.append({"k": k, "year": yr, "n": n, "up": up, "dn": dn,
                               "fk_up": fk_up, "fk_dn": fk_dn})

    # All-years combined
    tot = sum(r["n"] for r in combined_rows if r["k"] == k)
    ups = sum(r["up"] for r in combined_rows if r["k"] == k)
    dns = sum(r["dn"] for r in combined_rows if r["k"] == k)
    fup = sum(r["fk_up"] for r in combined_rows if r["k"] == k)
    fdn = sum(r["fk_dn"] for r in combined_rows if r["k"] == k)
    print(f"  {k:>4}  {'ALL':>5}  {tot:>8,}  {ups/tot*100:>6.1f}%  {dns/tot*100:>6.1f}%  "
          f"{(ups+dns)/tot*100:>6.1f}%  {fup/(fup+ups+1e-9)*100:>7.1f}%  {fdn/(fdn+dns+1e-9)*100:>7.1f}%")
    print()
