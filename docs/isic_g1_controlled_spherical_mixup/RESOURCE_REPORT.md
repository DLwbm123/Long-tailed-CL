# G1 resource report

{
  "wall_seconds": 44.60323952601175,
  "peak_RSS_bytes": 717590528,
  "observed_max_new_file_bytes": 70745132,
  "min_observed_free_bytes": 2265067520,
  "actual_formal_solve_calls": 108,
  "BLAS": [
    {
      "user_api": "blas",
      "internal_api": "openblas",
      "num_threads": 4,
      "prefix": "libopenblas",
      "filepath": "libopenblas64_p-r0-0cf96a72.3.23.dev.so",
      "version": "0.3.23.dev",
      "threading_layer": "pthreads",
      "architecture": "SapphireRapids"
    },
    {
      "user_api": "blas",
      "internal_api": "openblas",
      "num_threads": 4,
      "prefix": "libscipy_openblas",
      "filepath": "libscipy_openblas-6cdc3b4a.so",
      "version": "0.3.30",
      "threading_layer": "pthreads",
      "architecture": "SkylakeX"
    },
    {
      "user_api": "openmp",
      "internal_api": "openmp",
      "num_threads": 4,
      "prefix": "libgomp",
      "filepath": "libgomp-e985bcbb.so.1.0.0",
      "version": null
    }
  ],
  "neural_training_epochs": 0,
  "optimizer_steps": 0,
  "encoder_forward_calls": 0,
  "new_test_predictions": 0,
  "new_test_feature_reads": 0,
  "validation_adaptively_reused": true,
  "full_GSR_reproduction": false,
  "further_experiments_started": false,
  "core_logical_metric_rows": 108,
  "core_logical_per_class_rows": 648,
  "continued_increment_bytes_max": 4767808,
  "statistics_archives": "11 deduplicated final states plus one active state; no synthetic feature archives",
  "formal_augmented_stat_generations": 72,
  "separate_P0_head_probe_generations": 1,
  "source_commit": "4a7f8ff27fb46af748d354fbf6904780ff30c195",
  "technical_status": "COMPLETE_G1_P0_P3"
}

CPU only; one worker, four BLAS threads. Existing cache files were read in place. All generated data are on the same data mount; no historical file was deleted. File-size/free-space checks ran after each logical stage. Numerical kernels do not write temporary disk matrices. Aggregate reports and source snapshots are small additional overhead.

Final remote G1 directory including source/runtime overhead: 71,016,456 bytes (67.73 MiB). Conservative transient peak bound, adding one atomic state and 1 MiB metadata allowance: 73.27 MiB, below 256 MiB. Final free data-mount space: 2.109 GiB. Worker exit code 0, no worker remains. The execution timer covers audit/fitting/analysis (44.60 s), excluding code authoring, source transfer and publication; the separate toy precheck took 0.15 s.
