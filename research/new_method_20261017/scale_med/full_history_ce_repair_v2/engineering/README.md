# Engineering evidence and portable probes

LEGACY_COLD records a CPU-only reproduction of the legacy synthetic selfcheck under a fresh approved-NFS bytecode prefix, four threads, and the same interpreter. PRECHECK/PRECHECK_EXIT are the fixed cold-environment check and whole-child receipts; RECEIPT_TEST records forced-budget and injected-exception receipt checks. No real-data features, heads, or test inputs were read by these probes.

Portable probe sources differ from the executed counterparts only in private path binding: supply ENGINEERING_ROOT and LEGACY_SOURCE via the environment, and apply the environment.py runtime variables. This is not a claim of byte-for-byte identity or a new run. A forensic reproduction uses a fresh bytecode prefix; production precheck and RUN share the fixed runtime_cache prefix and exact saved environment.

BUDGET_LEDGER deducts the measured engineering phases plus a ten-second launch/metering reserve from the unchanged 1800-second wall and 3600-second CPU limits. Nested precheck child/initialization timings must not be added to coordinator wall time. The prior failed run lacks its own detailed inner fit receipt; this reproduction does not fabricate that missing history.
