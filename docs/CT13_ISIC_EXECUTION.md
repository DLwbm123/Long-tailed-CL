# CT13-ISIC execution boundary

`route_a/run_ct13.py --execute` is the real launcher. It writes a protocol
lock, audits the V2 train/val paths and the three normal Task1-end F1 parents,
and runs a fixed train-only feature qualification before any val access. The
validator (`--dry-run`) never starts this flow.

The launcher requires the exact locked `ViT-B-16/laion400m_e32` weights,
preprocess and tokenizer digests, plus read-only APART/CLIP factories. It
does not substitute OpenAI CLIP, regenerate a split, use an old 1536-D bank,
or open test/reserved manifests. If an asset is missing it writes
`ASSET_AUDIT.json`, `ENGINEERING_GATE.json`, and `BLOCKED.json` with the
checked paths and zero val/test/reserved access.

Once the gate passes, the factories must provide the locked text encoder and
moment-bank adapter for the fixed 2+2+2+2, three-order matrix. The launcher
does not select a seed, readout, prompt, or preprocessing rule from val.
