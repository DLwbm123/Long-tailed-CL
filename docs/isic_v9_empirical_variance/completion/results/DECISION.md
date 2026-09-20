# Stationary-feature experiment

{
  "status": "COMPLETE",
  "candidate": "K",
  "new_epochs": 60,
  "new_session_checkpoints": 6,
  "test_predictions": 0,
  "paired_real_streams_vs_E": "PASS",
  "contrasts": {
    "K-C": {
      "final_mean_deltas": {
        "balanced_accuracy": 9.647750173996194,
        "old_macro_recall": -4.482755610804394,
        "current_macro_recall": 52.039267528397964,
        "old_current_hm_macro_recall": 47.457542652803966,
        "tail_rank2": -4.754273504273503,
        "current_to_old_rate": -65.04318862752598,
        "old_to_current_rate": 19.42890136095991,
        "restricted_current_ba": 2.756498136932914
      },
      "seeds_improved_ba_and_current": 3,
      "new_zero_current": [],
      "success_criterion_pass": true
    },
    "K-E": {
      "final_mean_deltas": {
        "balanced_accuracy": 26.315239499226774,
        "old_macro_recall": 44.04299468323858,
        "current_macro_recall": -26.86802605280866,
        "old_current_hm_macro_recall": 40.77694206125157,
        "tail_rank2": 14.636752136752136,
        "current_to_old_rate": 32.14556639255434,
        "old_to_current_rate": -73.48008699112853,
        "restricted_current_ba": 0.7710888145670699
      },
      "seeds_improved_ba_and_current": 0,
      "new_zero_current": [],
      "success_criterion_pass": false
    },
    "K-F": {
      "final_mean_deltas": {
        "balanced_accuracy": 5.04057066822178,
        "old_macro_recall": 12.493444109297762,
        "current_macro_recall": -17.31804965500618,
        "old_current_hm_macro_recall": 3.5367250291148693,
        "tail_rank2": 10.04273504273504,
        "current_to_old_rate": 26.918467520877158,
        "old_to_current_rate": -24.745334724689997,
        "restricted_current_ba": 6.883932427410687
      },
      "seeds_improved_ba_and_current": 0,
      "new_zero_current": [],
      "success_criterion_pass": false
    },
    "K-G": {
      "final_mean_deltas": {
        "balanced_accuracy": 0.47957010145503887,
        "old_macro_recall": 6.573098768220713,
        "current_macro_recall": -17.801015898841985,
        "old_current_hm_macro_recall": -1.4469490586527105,
        "tail_rank2": 2.5106837606837615,
        "current_to_old_rate": 20.712532670363995,
        "old_to_current_rate": -12.404355197061854,
        "restricted_current_ba": 0.6642512077294688
      },
      "seeds_improved_ba_and_current": 0,
      "new_zero_current": [],
      "success_criterion_pass": false
    },
    "K-H": {
      "final_mean_deltas": {
        "balanced_accuracy": -1.0838949976012604,
        "old_macro_recall": 2.8265114850480657,
        "current_macro_recall": -12.815114445549233,
        "old_current_hm_macro_recall": -3.141226850249874,
        "tail_rank2": -2.884615384615382,
        "current_to_old_rate": 12.020940906483071,
        "old_to_current_rate": -7.4073478673698645,
        "restricted_current_ba": -2.5784145349362717
      },
      "seeds_improved_ba_and_current": 0,
      "new_zero_current": [],
      "success_criterion_pass": false
    },
    "K-I": {
      "final_mean_deltas": {
        "balanced_accuracy": 1.048695591378518,
        "old_macro_recall": 6.300759441003341,
        "current_macro_recall": -14.707495957495958,
        "old_current_hm_macro_recall": -0.3576463303274669,
        "tail_rank2": 2.0833333333333357,
        "current_to_old_rate": 15.845923376043856,
        "old_to_current_rate": -10.177930701663675,
        "restricted_current_ba": 0.1647759799933747
      },
      "seeds_improved_ba_and_current": 0,
      "new_zero_current": [],
      "success_criterion_pass": false
    },
    "K-J": {
      "final_mean_deltas": {
        "balanced_accuracy": -0.1467334503522224,
        "old_macro_recall": 4.885774611384363,
        "current_macro_recall": -15.244257635561985,
        "old_current_hm_macro_recall": -2.4846835034118655,
        "tail_rank2": -1.0149572649572651,
        "current_to_old_rate": 19.11933448078026,
        "old_to_current_rate": -12.286118083562874,
        "restricted_current_ba": 2.343334136812397
      },
      "seeds_improved_ba_and_current": 0,
      "new_zero_current": [],
      "success_criterion_pass": false
    }
  },
  "success": true
}

Development validation only. Full three-seed matrix; no epoch selection. Repeated validation use introduces adaptive selection bias. Frozen features do not eliminate Gaussian or augmentation mismatch. V2/V3 originals are unchanged.
