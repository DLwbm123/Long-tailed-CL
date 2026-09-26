# Medical VLM pretraining exposure audit

BiomedCLIP revision 9f341de24bfb00180f1b847274256e9b65a3a32e uses PMC-15M biomedical article figure-caption pairs. The released weight repository does not provide an exact training-image ID/hash index that can establish whether this project's ISIC/HK train/val images were exposed. PRETRAIN_EXPOSURE=UNKNOWN; no CLEAN claim is made. [Official model card](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224).

DermLIP revision 2509ca14a9971bdee4a1e35d8c107754fd158631 is trained on Derm1M. The official repository describes sources including public datasets, PubMed, medical forums, videos and teaching slides. This does not establish sample-level separation from ISIC. The [Derm1M index](https://huggingface.co/datasets/redlessone/Derm1M) requires accepting access conditions and sharing contact information. No exact training index was available for a lawful ID/hash intersection. Existing project train/val locks remain unchanged. Overlap=UNKNOWN; NOT_MEASURED is not zero overlap.

The current DermLIP card metadata says CC BY-NC 4.0, but its body and gated terms say CC BY-NC-ND 4.0. No separate weight-license file is listed. Anonymous model assets return HTTP 401; the existing account's access check returns HTTP 403. D is BLOCKED_ACCESS_LICENSE and does not affect the G/B primary gate. No alternate copy or older revision was used. [Official model card](https://huggingface.co/redlessone/DermLIP_ViT-B-16).

No sealed test/reserved images, features, predictions or identity maps were opened for this audit. No new model was selected using validation results.
