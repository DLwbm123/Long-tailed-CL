# Stationary-feature experiment

{
  "status": "COMPLETE",
  "candidate": "I",
  "new_epochs": 60,
  "new_session_checkpoints": 6,
  "test_predictions": 0,
  "paired_real_streams_vs_E": "PASS",
  "contrasts": {
    "I-C": {
      "final_mean_deltas": {
        "balanced_accuracy": 8.599054582617676,
        "old_macro_recall": -10.783515051807735,
        "current_macro_recall": 66.74676348589392,
        "old_current_hm_macro_recall": 47.815188983131435,
        "tail_rank2": -6.837606837606839,
        "current_to_old_rate": -80.88911200356983,
        "old_to_current_rate": 29.606832062623585,
        "restricted_current_ba": 2.5917221569395394
      },
      "seeds_improved_ba_and_current": 3,
      "new_zero_current": [],
      "success_criterion_pass": false
    },
    "I-E": {
      "final_mean_deltas": {
        "balanced_accuracy": 25.266543907848256,
        "old_macro_recall": 37.74223524223524,
        "current_macro_recall": -12.160530095312707,
        "old_current_hm_macro_recall": 41.13458839157904,
        "tail_rank2": 12.5534188034188,
        "current_to_old_rate": 16.299643016510487,
        "old_to_current_rate": -63.302156289464854,
        "restricted_current_ba": 0.6063128345736951
      },
      "seeds_improved_ba_and_current": 0,
      "new_zero_current": [],
      "success_criterion_pass": false
    },
    "I-F": {
      "final_mean_deltas": {
        "balanced_accuracy": 3.9918750768432623,
        "old_macro_recall": 6.192684668294421,
        "current_macro_recall": -2.6105536975102246,
        "old_current_hm_macro_recall": 3.8943713594423364,
        "tail_rank2": 7.959401709401706,
        "current_to_old_rate": 11.072544144833302,
        "old_to_current_rate": -14.567404023026326,
        "restricted_current_ba": 6.719156447417312
      },
      "seeds_improved_ba_and_current": 1,
      "new_zero_current": [],
      "success_criterion_pass": false
    },
    "I-G": {
      "final_mean_deltas": {
        "balanced_accuracy": -0.5691254899234792,
        "old_macro_recall": 0.27233932721737136,
        "current_macro_recall": -3.093519941346029,
        "old_current_hm_macro_recall": -1.0893027283252437,
        "tail_rank2": 0.42735042735042583,
        "current_to_old_rate": 4.866609294320138,
        "old_to_current_rate": -2.2264244953981795,
        "restricted_current_ba": 0.49947522773609404
      },
      "seeds_improved_ba_and_current": 1,
      "new_zero_current": [],
      "success_criterion_pass": false
    },
    "I-H": {
      "final_mean_deltas": {
        "balanced_accuracy": -2.1325905889797787,
        "old_macro_recall": -3.4742479559552755,
        "current_macro_recall": 1.892381511946726,
        "old_current_hm_macro_recall": -2.7835805199224075,
        "tail_rank2": -4.967948717948718,
        "current_to_old_rate": -3.824982469560785,
        "old_to_current_rate": 2.7705828342938097,
        "restricted_current_ba": -2.7431905149296463
      },
      "seeds_improved_ba_and_current": 0,
      "new_zero_current": [],
      "success_criterion_pass": false
    }
  },
  "success": false
}

Development validation only. Full three-seed matrix; no epoch selection. Repeated validation use introduces adaptive selection bias. Frozen features do not eliminate Gaussian or augmentation mismatch. V2/V3 originals are unchanged.
