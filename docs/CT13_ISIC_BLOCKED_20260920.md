# CT13-ISIC execution record (blocked at asset gate)

The real launcher was executed on `my-gpu` at
`/remote-home/wangbomin/LongTailedCL/ct13_runs/isic_20260920` from the CT13
source checkout. It read the V2 **train** manifest and checked the val path
for existence only. It did not open val rows, test/reserved manifests,
images from val/test, or any test prediction path.

The audit found the actual V2 ISIC image directory, 18,718 retained train
rows, and all three normal Task1-end F1 parent states (`U_t01`) with their
sidecar lock records. The train image paths were present. The run stopped
before engineering qualification and before any statistic fitting or val
evaluation because the exact locked `ViT-B-16/laion400m_e32` CLIP weights,
preprocess/tokenizer assets, and their digests were not present in the
configured/search locations. The V2 summary also reports
`semantic_map: SEMANTIC_MAP_UNVERIFIED`; no class-name mapping was silently
invented for text prompts.

The launcher wrote `ASSET_AUDIT.json`, `ENGINEERING_GATE.json`,
`PROTOCOL_LOCK.json`, and `BLOCKED.json`. The receipts record
`val_access=0`, `test_access=0`, `reserved_access=0`, zero training epochs and
zero optimizer steps. This is a concrete asset block, not `CONFIG_VALID` and
not a substitute-model run.

Minimum resume condition: provide the exact locked CLIP weight file plus
preprocess/tokenizer files and SHA256 values, and a verified eight-class name
mapping. The existing F1 parent states remain preserved and reusable. No
CT13 matrix or metrics are reported until that gate passes.
