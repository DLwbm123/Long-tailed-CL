# Reused frozen dependencies

Source-only predecessor: producer_group_crossfit at scientific source3d5eff2015b504621ff5670b78b165b5145088ba. Reuse train_base.py (original APART setup/raw/guard), binding.py, geometry.py (fixed mapper application only, no teacher solve), data.py (initial pixels filtering), and space.py. Do not modify those source files or invoke predecessor TRAIN/EVALUATE. Runtime paths supplied only in private config. No new dependencies/model download/environment copy.

Run entry has no optimizer and never calls teacher() or any ridge solve. Runtime install_guard plus explicit old-window allowlist permits only predecessor prep PLAN, two original F teacher files and source modules; all learned checkpoints and earlier risk tables are prohibited. Public outputs aggregate the8cells, not group rows or ordered group metrics.
