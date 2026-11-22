"""Put refit bootstrap intervals on the contemporary (2018-2024) estimands.

The two contemporary tables were written by ``run_fulldata_recentdecade_paf*.py``,
whose bootstrap resampled countries while holding the fitted surface fixed. Their
intervals therefore reflected exposure composition but not coefficient uncertainty,
while the Methods state that every replicate refits the model. The 2026-09 reviewer
round re-ran both surfaces with a bootstrap that refits on every replicate
(``sens2026_survival`` / ``sens2026_binary``, model ``main``; same seed, same draws as
the long-window intervals) and reported them as a robustness check.

Since 2026-11-17 (supervisor review, point on bootstrap intervals) the refit intervals
are the reported ones. This module copies them into the two contemporary tables, so
every consumer -- the burden reconciliation, Table 2, the IGME-uncertainty eTable,
the registry -- reads one set. The fixed-surface intervals are kept alongside as
``ci_lo_fixed_surface`` / ``ci_hi_fixed_surface``. Point estimates are not touched:
the script refuses to run if the refit runs' points differ from the tables'.

Idempotent. Run:  python -m src.inferential.contemporary_refit_ci
"""
from __future__ import annotations

import pandas as pd

from config.paths import OUTPUT_DIR

MV = OUTPUT_DIR / "phase4" / "manuscript"
SENS = OUTPUT_DIR / "phase4" / "sensitivity_2026-09"
TABLES = {
    "table_recentdecade_paf_survival.csv": SENS / "survival_main" / "summary.csv",
    "table_recentdecade_paf_reweighted.csv": SENS / "binary_main" / "summary.csv",
}
METHOD = ("country-cluster percentile bootstrap, 1000 replicates, model refitted in every "
          "replicate (sens2026 main)")


def main() -> None:
    for table_name, summary_path in TABLES.items():
        path = MV / table_name
        table = pd.read_csv(path)
        refit = (pd.read_csv(summary_path).query("estimand == 'contemporary'")
                 .set_index("scenario"))
        if "ci_lo_fixed_surface" not in table.columns:
            table["ci_lo_fixed_surface"] = table["ci_lo"]
            table["ci_hi_fixed_surface"] = table["ci_hi"]
        for i, row in table.iterrows():
            r = refit.loc[row["scenario"]]
            if abs(float(r["paf_pct"]) - float(row["paf_recent_prevalence_pct"])) > 0.05:
                raise SystemExit(f"{table_name} {row['scenario']}: refit point {r['paf_pct']} "
                                 f"differs from {row['paf_recent_prevalence_pct']}")
            table.loc[i, "ci_lo"] = float(r["ci_lo"])
            table.loc[i, "ci_hi"] = float(r["ci_hi"])
        table["ci_method"] = METHOD
        table.to_csv(path, index=False)
        print(f"{table_name}:")
        print(table[["scenario", "paf_recent_prevalence_pct", "ci_lo", "ci_hi",
                     "ci_lo_fixed_surface", "ci_hi_fixed_surface"]].to_string(index=False))


if __name__ == "__main__":
    main()
