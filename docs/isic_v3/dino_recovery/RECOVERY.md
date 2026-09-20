# DINOv2 availability recovery

The user authorized local/my-gpu download and transfer to the active compute server. Direct connections from both hosts failed; the fixed publisher revision was downloaded successfully through the existing local proxy, without changing proxy settings.

The 346,345,912-byte model.safetensors matches the approved SHA256 d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841 on the local download and the compute-server copy. config.json and preprocessor_config.json come from the same revision; exact hashes are in VERIFIED_WEIGHT_LOCK.json.

Standard Dinov2Model CPU loading passed with no missing, unexpected, mismatched keys or error messages. Shape is 768 dimensions / patch14 / float32. Dependencies are installed in a separate supplemental directory; the active P2 environment and frozen source are unchanged. Loading did not forward images.

The supplemental queue waits for the original P2 supervisor to finish and a free GPU process slot. It then reuses the locked P1 implementation for B only: one train/val feature extraction, F-NCM and fixed class-balanced CBRidge, no test features or predictions. A artifacts are reused without extracting or fitting again. Both-encoder reports and the combined decision are written in a new output directory; original partial reports and locks remain intact. The 15-minute supplemental reserve plus the existing conservative budget remains below eight GPU hours. No additional training epochs or trajectories are introduced.

This report records verified transfer/loading and queued baseline completion, not completed DINOv2 performance. The supplement stops after the requested P0/P1/P2 combined report. No P3 or extra methods are scheduled.
