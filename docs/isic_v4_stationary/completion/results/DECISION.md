# Stationary-feature experiment

{
  "status": "COMPLETE",
  "new_epochs": 60,
  "new_session_checkpoints": 6,
  "test_predictions": 0,
  "paired_real_streams_vs_E": "PASS",
  "contrasts": {
    "F-C": {
      "final_mean_deltas": {
        "balanced_accuracy": 4.607179505774414,
        "old_macro_recall": -16.976199720102155,
        "current_macro_recall": 69.35731718340413,
        "old_current_hm_macro_recall": 43.920817623689096,
        "tail_rank2": -14.797008547008545,
        "current_to_old_rate": -91.96165614840312,
        "old_to_current_rate": 44.17423608564991,
        "restricted_current_ba": -4.1274342904777725
      },
      "seeds_improved_ba_and_current": 2,
      "new_zero_current": [],
      "success_criterion_pass": false
    },
    "F-E": {
      "final_mean_deltas": {
        "balanced_accuracy": 21.27466883100499,
        "old_macro_recall": 31.54955057394082,
        "current_macro_recall": -9.549976397802482,
        "old_current_hm_macro_recall": 37.240217032136705,
        "tail_rank2": 4.594017094017095,
        "current_to_old_rate": 5.2270988716771845,
        "old_to_current_rate": -48.734752266438534,
        "restricted_current_ba": -6.1128436128436165
      },
      "seeds_improved_ba_and_current": 1,
      "new_zero_current": [],
      "success_criterion_pass": false
    }
  },
  "success": false
}

Development validation only. Full three-seed matrix; no epoch selection. Repeated validation use introduces adaptive selection bias. Frozen features do not eliminate Gaussian or augmentation mismatch. V2/V3 originals are unchanged.
