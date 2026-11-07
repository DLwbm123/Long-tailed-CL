# R112 evidence memory

See [PROTOCOL_NEW.md](PROTOCOL_NEW.md), [BYTE_LEDGER.json](BYTE_LEDGER.json), and [SOURCE_SUPPLY_RESULT.json](SOURCE_SUPPLY_RESULT.json). This is a bounded protocol repair, with conditional real classifier branches and no controller/RL/VAL/TEST. Runtime phase status is reported separately from frozen design.

Code imports the fixed sibling R110 learner and paired update law, and R111 query-free ZERO construction. Private configs provide legal TRAIN assets and neutral runtime paths. `main.py` loads Q_SOURCE/Q_MODULE; `control.py` independently guards construction, full native admission and science. `selfcheck.py` is synthetic only.
