import logging
import json
import numpy as np
import os
import random
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import AdapterVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, accuracy


num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)   
        self._network = AdapterVitNet(args, True)

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.theta = args["theta"] if args["theta"] is not None else 100
        self.args = args
        self.concm_stage1_enabled = bool(args.get("concm_stage1", False))
        self.concm_stage1_loss_weight = float(args.get("concm_stage1_loss_weight", 0.0))
        self.concm_stage1_synth_per_class = int(args.get("concm_stage1_synth_per_class", 4))
        self.concm_stage1_cov_eps = float(args.get("concm_stage1_cov_eps", 1e-6))
        self.concm_stage1_var_max = float(args.get("concm_stage1_var_max", 1.0))
        self.concm_stage1_start_epoch = int(args.get("concm_stage1_start_epoch", 1))
        self.concm_stage1_ramp_epochs = int(args.get("concm_stage1_ramp_epochs", 0))
        self.concm_stage1_max_synth_total = int(args.get("concm_stage1_max_synth_total", 0))
        self.concm_replay_bias_controller = bool(args.get("concm_replay_bias_controller", False))
        self.concm_replay_target_bias = float(args.get("concm_replay_target_bias", 0.12))
        self.concm_replay_min_scale = float(args.get("concm_replay_min_scale", 0.5))
        self.concm_replay_max_scale = float(args.get("concm_replay_max_scale", 1.0))
        self.concm_replay_bias_eps = float(args.get("concm_replay_bias_eps", 1e-6))
        self.concm_stage1_memory = {}
        self._previous_eval_top1 = None
        self.concm_stage1_eval_calibration = bool(args.get("concm_stage1_eval_calibration", False))
        self.calibration_rule = args.get("calibration_rule", "none")
        self._last_eval_calibration = None
        self.concm_proto_calibration = bool(args.get("concm_proto_calibration", False))
        self.concm_proto_topk = int(args.get("concm_proto_topk", 5))
        self.concm_alpha_a = float(args.get("concm_alpha_a", 2.0))
        self.concm_alpha_b = float(args.get("concm_alpha_b", 1.0))
        self.concm_alpha_min = float(args.get("concm_alpha_min", 0.15))
        self.concm_alpha_max = float(args.get("concm_alpha_max", 0.95))
        self._last_proto_calibration_diag = {}
        self.concm_use_match_loss = bool(args.get("concm_use_match_loss", False))
        self.concm_match_lambda = float(args.get("concm_match_lambda", 0.0))
        self.concm_match_warmup_epoch = int(args.get("concm_match_warmup_epoch", 5))
        self.concm_match_paths = args.get("concm_match_paths", "both")
        self.concm_match_detach_anchors = bool(args.get("concm_match_detach_anchors", True))
        self.concm_match_current_only = bool(args.get("concm_match_current_only", True))
        self.concm_match_tail_weight = bool(args.get("concm_match_tail_weight", False))
        self.concm_match_anchors = None
        self._last_match_anchor_diag = {}
        self.concm_use_uncertainty_selective_match = bool(args.get("concm_use_uncertainty_selective_match", False))
        self.concm_usfm_lambda = float(args.get("concm_usfm_lambda", 0.0))
        self.concm_usfm_warmup_epoch = int(args.get("concm_usfm_warmup_epoch", 5))
        self.concm_usfm_paths = args.get("concm_usfm_paths", "both")
        self.concm_usfm_current_real_only = bool(args.get("concm_usfm_current_real_only", True))
        self.concm_usfm_detach_gate = bool(args.get("concm_usfm_detach_gate", True))
        self.concm_usfm_detach_anchors = bool(args.get("concm_usfm_detach_anchors", True))
        self.concm_usfm_anchor_momentum = float(args.get("concm_usfm_anchor_momentum", 0.9))
        self.concm_usfm_gate_k = float(args.get("concm_usfm_gate_k", 1.0))
        self.concm_usfm_anchors = None
        self._last_usfm_anchor_diag = {}
        self.concm_uncertainty_diagnostics = bool(args.get("concm_uncertainty_diagnostics", False))
        self.concm_uncertainty_temp = float(args.get("concm_uncertainty_temp", 1.0))
        self.concm_uncertainty_ema_momentum = float(args.get("concm_uncertainty_ema_momentum", 0.9))
        self.concm_uncertainty_gated_replay = bool(args.get("concm_uncertainty_gated_replay", False))
        self.concm_ugr_min_scale = float(args.get("concm_ugr_min_scale", 0.5))
        self.concm_ugr_max_scale = float(args.get("concm_ugr_max_scale", 1.0))
        self.concm_ugr_target_ale = float(args.get(
            "concm_ugr_target_ale",
            args.get("concm_ugr_target_uncertainty", 0.6),
        ))
        self.concm_ugr_eps = float(args.get("concm_ugr_eps", 1e-6))
        self.concm_ugr_use_batch_current_only = bool(args.get("concm_ugr_use_batch_current_only", True))
        self._uncertainty_ema_ale = None
        self._uncertainty_ema_epi = None
        self._uncertainty_ema_count = None
        self._last_uncertainty_diag = {}

        # Freeze the parameters for ViT.
        if self.args["freeze"]:
            for p in self._network.original_backbone.parameters():
                p.requires_grad = False
        
        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} model training parameters.')

        # if some parameters are trainable, print the key name and corresponding parameter number
        if total_params != total_trainable_params:
            for name, param in self._network.backbone.named_parameters():
                if param.requires_grad:
                    logging.info("{}: {}".format(name, param.numel()))

    def after_task(self):
        self._known_classes = self._total_classes       

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        if self.concm_stage1_enabled:
            self._update_concm_stage1_memory()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self.args["imbalance"]:
            cls_num_list = torch.Tensor(self.args["lt_list"][:self._total_classes]).to(self._device)
        
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
            
        if self._cur_task > 0:
            self._init_prompt(optimizer)

        if self._cur_task > 0 and self.args["reinit_optimizer"]: # true
            optimizer = self.get_optimizer()
        self._reset_concm_usfm_anchors()
        self._prepare_concm_match_anchors()
        self._init_train(train_loader, test_loader, optimizer, scheduler)

    def get_optimizer(self):
        net_param = [p for name, p in self._network.backbone.named_parameters() if  p.requires_grad]
        
        base_param =  [p for name, p in self._network.backbone.named_parameters() if 'pool' in name and p.requires_grad]
        fc_param =  [p for name, p in self._network.backbone.named_parameters() if 'pool' not in name and p.requires_grad]
        
        param_f = {'params': base_param, 'lr': self.init_lr * 0.1, 'weight decay': self.weight_decay}
        param_s = {'params': fc_param, 'lr': self.init_lr, 'weight decay': self.weight_decay}
        param = [param_f, param_s]

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                param
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                param
            )
            
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                param
            )

        return optimizer
    
    def get_scheduler(self, optimizer):
        if self.args["init_cls"]  != self.args["increment"]:
            self.args["scheduler"] = "cosine"
        else:
            self.args["scheduler"] = "constant"

        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_prompt(self, optimizer):
        args = self.args
        model = self._network.backbone
        task_id = self._cur_task

        # Transfer previous learned prompt params to the new prompt
        if args["prompt_pool"] and args["shared_prompt_pool"]:
            prev_start = (task_id - 1) * args["pool_size"]
            prev_end = task_id * args["pool_size"]

            cur_start = prev_end
            cur_end = (task_id + 1) * args["pool_size"]

            if (prev_end > args["size"]) or (cur_end > args["size"]):
                pass
            else:
                cur_idx = (slice(cur_start, cur_end))
                prev_idx = (slice(prev_start, prev_end))
                
        # Transfer previous learned prompt param keys to the new prompt
        if args["prompt_pool"] and args["shared_prompt_key"]:
            prev_start = (task_id - 1) * args["pool_size"]
            prev_end = task_id * args["pool_size"]

            cur_start = prev_end
            cur_end = (task_id + 1) * args["pool_size"]

            if (prev_end > args["size"]) or (cur_end > args["size"]):
                pass
            else:
                cur_idx = (slice(cur_start, cur_end))
                prev_idx = (slice(prev_start, prev_end))

    def _backbone_module(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.backbone
        return self._network.backbone

    def _concm_stage1_effective_weight(self, epoch):
        if (
            not self.concm_stage1_enabled
            or self.concm_stage1_loss_weight <= 0
            or len(self.concm_stage1_memory) == 0
            or self._known_classes == 0
        ):
            return 0.0

        epoch_one_based = epoch + 1
        if epoch_one_based < self.concm_stage1_start_epoch:
            return 0.0

        if self.concm_stage1_ramp_epochs > 0:
            if self.concm_stage1_ramp_epochs == 1:
                ramp = 1.0
            else:
                ramp_pos = epoch_one_based - self.concm_stage1_start_epoch
                ramp = min(max(ramp_pos / float(self.concm_stage1_ramp_epochs - 1), 0.0), 1.0)
            return self.concm_stage1_loss_weight * ramp

        return self.concm_stage1_loss_weight

    def _concm_replay_controller_state(self, all_seen_logits):
        if (
            not self.concm_replay_bias_controller
            or all_seen_logits is None
            or self._cur_task <= 0
            or self._known_classes <= 0
        ):
            return None

        with torch.no_grad():
            preds = all_seen_logits.detach().argmax(dim=1)
            old_rate = (preds < self._known_classes).float().mean().item()

        replay_scale = self.concm_replay_target_bias / (old_rate + self.concm_replay_bias_eps)
        replay_scale = min(max(replay_scale, self.concm_replay_min_scale), self.concm_replay_max_scale)
        if not np.isfinite(replay_scale):
            replay_scale = self.concm_replay_min_scale

        effective_synth_cap = self.concm_stage1_max_synth_total
        if self.concm_stage1_max_synth_total > 0:
            effective_synth_cap = int(round(self.concm_stage1_max_synth_total * replay_scale))
            effective_synth_cap = max(1, min(self.concm_stage1_max_synth_total, effective_synth_cap))

        return {
            "train_current_pred_old_rate": old_rate * 100.0,
            "replay_scale": replay_scale,
            "effective_synth_cap": effective_synth_cap,
        }

    def _concm_uncertainty_gated_replay_state(self, ale, epi, all_seen_logits, targets):
        if (
            not self.concm_uncertainty_gated_replay
            or ale is None
            or epi is None
            or all_seen_logits is None
            or self._cur_task <= 0
            or self._known_classes <= 0
        ):
            return None

        with torch.no_grad():
            if self.concm_ugr_use_batch_current_only:
                mask = torch.logical_and(targets >= self._known_classes, targets < self._total_classes)
            else:
                mask = torch.ones_like(targets, dtype=torch.bool)
            if not mask.any():
                return None

            ale_current = ale.detach()[mask]
            epi_current = epi.detach()[mask]
            batch_ale_current = ale_current.mean().item()
            batch_epi_current = epi_current.mean().item()
            preds = all_seen_logits.detach()[mask].argmax(dim=1)
            current_pred_old_rate = (preds < self._known_classes).float().mean().item()

        replay_scale = self.concm_ugr_target_ale / (batch_ale_current + self.concm_ugr_eps)
        replay_scale = min(max(replay_scale, self.concm_ugr_min_scale), self.concm_ugr_max_scale)
        if not np.isfinite(replay_scale):
            replay_scale = self.concm_ugr_min_scale

        effective_synth_cap = self.concm_stage1_max_synth_total
        if self.concm_stage1_max_synth_total > 0:
            effective_synth_cap = int(round(self.concm_stage1_max_synth_total * replay_scale))
            effective_synth_cap = max(1, min(self.concm_stage1_max_synth_total, effective_synth_cap))

        return {
            "batch_Ale_current": batch_ale_current,
            "batch_Epi_current": batch_epi_current,
            "train_current_pred_old_rate": current_pred_old_rate * 100.0,
            "replay_scale": replay_scale,
            "effective_synth_cap": effective_synth_cap,
        }

    def _concm_stage1_sample_memory(self, max_synth_total=None):
        if (
            not self.concm_stage1_enabled
            or self.concm_stage1_synth_per_class <= 0
            or len(self.concm_stage1_memory) == 0
            or self._known_classes == 0
        ):
            return None

        main_features, few_features, labels = [], [], []
        for class_id in sorted(self.concm_stage1_memory):
            if class_id >= self._known_classes:
                continue
            stats = self.concm_stage1_memory[class_id]
            mean_main = stats["mean_main"].to(self._device)
            var_main = stats["var_main"].to(self._device)
            mean_few = stats["mean_few"].to(self._device)
            var_few = stats["var_few"].to(self._device)

            count = self.concm_stage1_synth_per_class
            std_main = torch.sqrt(torch.clamp(var_main, 0, self.concm_stage1_var_max) + self.concm_stage1_cov_eps)
            std_few = torch.sqrt(torch.clamp(var_few, 0, self.concm_stage1_var_max) + self.concm_stage1_cov_eps)
            main_features.append(mean_main.unsqueeze(0) + torch.randn(count, mean_main.numel(), device=self._device) * std_main.unsqueeze(0))
            few_features.append(mean_few.unsqueeze(0) + torch.randn(count, mean_few.numel(), device=self._device) * std_few.unsqueeze(0))
            labels.append(torch.full((count,), class_id, dtype=torch.long, device=self._device))

        if len(labels) == 0:
            return None

        main_features = torch.cat(main_features, dim=0)
        few_features = torch.cat(few_features, dim=0)
        labels = torch.cat(labels, dim=0)
        synth_cap = self.concm_stage1_max_synth_total if max_synth_total is None else int(max_synth_total)
        if synth_cap > 0 and labels.numel() > synth_cap:
            selected = torch.randperm(labels.numel(), device=self._device)[:synth_cap]
            main_features = main_features[selected]
            few_features = few_features[selected]
            labels = labels[selected]
        return main_features, few_features, labels

    def _concm_stage1_loss(self, max_synth_total=None):
        sampled = self._concm_stage1_sample_memory(max_synth_total=max_synth_total)
        if sampled is None:
            return None

        main_features, few_features, labels = sampled
        backbone = self._backbone_module()
        logits = backbone.head(main_features)[:, : self._total_classes]
        logits_few = backbone.head_few(few_features)[:, : self._total_classes]
        return F.cross_entropy(logits + logits_few, labels)

    def _entropy_from_probs(self, probs):
        probs = probs.clamp_min(1e-8)
        return -(probs * probs.log()).sum(dim=1)

    def _concm_uncertainty_from_logits(self, logits_main, logits_few):
        if not (
            self.concm_uncertainty_diagnostics
            or self.concm_uncertainty_gated_replay
            or self.concm_use_uncertainty_selective_match
        ):
            return None

        temp = max(self.concm_uncertainty_temp, 1e-6)
        with torch.no_grad():
            p_main = F.softmax(logits_main.detach() / temp, dim=1)
            p_few = F.softmax(logits_few.detach() / temp, dim=1)
            p_bar = 0.5 * (p_main + p_few)
            ale = 0.5 * (self._entropy_from_probs(p_main) + self._entropy_from_probs(p_few))
            epi = (self._entropy_from_probs(p_bar) - ale).clamp_min(0.0)
        return ale, epi

    def _ensure_uncertainty_ema(self):
        if self._uncertainty_ema_ale is not None:
            return
        num_classes = int(self.args.get("nb_classes", self._total_classes))
        self._uncertainty_ema_ale = torch.zeros(num_classes)
        self._uncertainty_ema_epi = torch.zeros(num_classes)
        self._uncertainty_ema_count = torch.zeros(num_classes)

    def _update_uncertainty_ema(self, ale, epi, targets):
        if not self.concm_uncertainty_diagnostics:
            return
        self._ensure_uncertainty_ema()
        ale_cpu = ale.detach().cpu()
        epi_cpu = epi.detach().cpu()
        targets_cpu = targets.detach().cpu().long()
        momentum = self.concm_uncertainty_ema_momentum
        with torch.no_grad():
            for class_id in targets_cpu.unique():
                class_id = int(class_id.item())
                mask = targets_cpu == class_id
                ale_mean = ale_cpu[mask].mean()
                epi_mean = epi_cpu[mask].mean()
                if self._uncertainty_ema_count[class_id] > 0:
                    self._uncertainty_ema_ale[class_id] = momentum * self._uncertainty_ema_ale[class_id] + (1.0 - momentum) * ale_mean
                    self._uncertainty_ema_epi[class_id] = momentum * self._uncertainty_ema_epi[class_id] + (1.0 - momentum) * epi_mean
                else:
                    self._uncertainty_ema_ale[class_id] = ale_mean
                    self._uncertainty_ema_epi[class_id] = epi_mean
                self._uncertainty_ema_count[class_id] += 1

    def _uncertainty_mask_stats(self, name, mask, ale, epi):
        if mask is None or not np.any(mask):
            return {
                "{}_n".format(name): 0,
                "{}_ale".format(name): None,
                "{}_epi".format(name): None,
            }
        return {
            "{}_n".format(name): int(mask.sum()),
            "{}_ale".format(name): float(np.around(ale[mask].mean(), decimals=6)),
            "{}_epi".format(name): float(np.around(epi[mask].mean(), decimals=6)),
        }

    def _safe_corr(self, x, y):
        if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
            return None
        return float(np.around(np.corrcoef(x, y)[0, 1], decimals=6))

    def _build_uncertainty_eval_diag(self, labels, calibrated_preds, uncalibrated_preds, ale, epi):
        labels = labels.astype(np.int64)
        calibrated_preds = calibrated_preds.astype(np.int64)
        uncalibrated_preds = uncalibrated_preds.astype(np.int64)
        ale = ale.astype(np.float64)
        epi = epi.astype(np.float64)

        cls_num_list = np.array(self.args["lt_list"][: self._total_classes])
        label_counts = cls_num_list[labels]
        old_mask = labels < self._known_classes if self._known_classes > 0 else np.zeros_like(labels, dtype=bool)
        new_mask = labels >= self._known_classes if self._known_classes > 0 else np.ones_like(labels, dtype=bool)
        many_mask = label_counts > 100
        few_mask = label_counts < 20
        medium_mask = np.logical_not(np.logical_or(many_mask, few_mask))
        correct_mask = calibrated_preds == labels
        incorrect_mask = np.logical_not(correct_mask)
        new_pred_old_mask = np.logical_and(new_mask, calibrated_preds < self._known_classes) if self._known_classes > 0 else np.zeros_like(labels, dtype=bool)
        old_pred_new_mask = np.logical_and(old_mask, calibrated_preds >= self._known_classes) if self._known_classes > 0 else np.zeros_like(labels, dtype=bool)
        uncalibrated_correct_mask = uncalibrated_preds == labels
        calibrated_changed_mask = calibrated_preds != uncalibrated_preds

        diag = {}
        for name, mask in (
            ("unc_old", old_mask),
            ("unc_new", new_mask),
            ("unc_many", many_mask),
            ("unc_medium", medium_mask),
            ("unc_few", few_mask),
            ("unc_correct", correct_mask),
            ("unc_incorrect", incorrect_mask),
            ("unc_new_pred_old", new_pred_old_mask),
            ("unc_old_pred_new", old_pred_new_mask),
            ("unc_uncalibrated_correct", uncalibrated_correct_mask),
            ("unc_uncalibrated_incorrect", np.logical_not(uncalibrated_correct_mask)),
            ("unc_calibration_changed", calibrated_changed_mask),
        ):
            diag.update(self._uncertainty_mask_stats(name, mask, ale, epi))

        error = incorrect_mask.astype(np.float64)
        uncalibrated_error = np.logical_not(uncalibrated_correct_mask).astype(np.float64)
        diag["unc_epi_error_corr"] = self._safe_corr(epi, error)
        diag["unc_ale_error_corr"] = self._safe_corr(ale, error)
        diag["unc_epi_uncalibrated_error_corr"] = self._safe_corr(epi, uncalibrated_error)
        diag["unc_ale_uncalibrated_error_corr"] = self._safe_corr(ale, uncalibrated_error)
        diag["unc_nan_or_inf"] = int(
            np.isnan(ale).any()
            or np.isnan(epi).any()
            or np.isinf(ale).any()
            or np.isinf(epi).any()
        )

        self._log_uncertainty_eval_diag(diag)
        return diag

    def _log_uncertainty_eval_diag(self, diag):
        if not self.concm_uncertainty_diagnostics:
            return
        split_names = [
            "unc_old",
            "unc_new",
            "unc_many",
            "unc_medium",
            "unc_few",
            "unc_correct",
            "unc_incorrect",
            "unc_new_pred_old",
            "unc_old_pred_new",
            "unc_uncalibrated_correct",
            "unc_uncalibrated_incorrect",
            "unc_calibration_changed",
        ]
        parts = []
        for name in split_names:
            n = diag.get("{}_n".format(name), 0)
            ale = diag.get("{}_ale".format(name))
            epi = diag.get("{}_epi".format(name))
            if ale is None or epi is None:
                parts.append("{}:n={}".format(name, n))
            else:
                parts.append("{}:n={},Ale={:.6f},Epi={:.6f}".format(name, n, ale, epi))
        logging.info("ConCMUncertaintyEval task={} {}".format(self._cur_task, " | ".join(parts)))
        logging.info(
            "ConCMUncertaintyEvalCorr task={} epi_error_corr={} ale_error_corr={} "
            "epi_uncalibrated_error_corr={} ale_uncalibrated_error_corr={} nan_or_inf={}".format(
                self._cur_task,
                diag.get("unc_epi_error_corr"),
                diag.get("unc_ale_error_corr"),
                diag.get("unc_epi_uncalibrated_error_corr"),
                diag.get("unc_ale_uncalibrated_error_corr"),
                diag.get("unc_nan_or_inf", 0),
            )
        )

    def _uncertainty_ema_diag(self):
        if self._uncertainty_ema_ale is None:
            return {}
        seen = np.arange(0, self._total_classes)
        counts = np.array(self.args["lt_list"][: self._total_classes])
        active = tensor2numpy(self._uncertainty_ema_count[: self._total_classes]) > 0
        if not active.any():
            return {}
        ale = tensor2numpy(self._uncertainty_ema_ale[: self._total_classes])
        epi = tensor2numpy(self._uncertainty_ema_epi[: self._total_classes])
        old_mask = seen < self._known_classes if self._known_classes > 0 else np.zeros_like(seen, dtype=bool)
        new_mask = seen >= self._known_classes if self._known_classes > 0 else np.ones_like(seen, dtype=bool)
        many_mask = counts > 100
        few_mask = counts < 20
        medium_mask = np.logical_not(np.logical_or(many_mask, few_mask))
        diag = {}
        for name, mask in (
            ("unc_ema_old", old_mask),
            ("unc_ema_new", new_mask),
            ("unc_ema_many", many_mask),
            ("unc_ema_medium", medium_mask),
            ("unc_ema_few", few_mask),
        ):
            mask = np.logical_and(mask, active)
            diag.update(self._uncertainty_mask_stats(name, mask, ale, epi))
        return diag

    def _concm_match_paths(self):
        paths = str(self.concm_match_paths).lower().replace("+", ",").split(",")
        paths = [path.strip() for path in paths if path.strip()]
        if "both" in paths:
            return ["main", "few"]
        valid = []
        for path in paths:
            if path not in ("main", "few"):
                raise ValueError("Unsupported concm_match_paths: {}".format(self.concm_match_paths))
            valid.append(path)
        return valid

    def _concm_match_effective_weight(self, epoch):
        if (
            not self.concm_use_match_loss
            or self.concm_match_lambda <= 0
            or self._cur_task <= 0
            or self._known_classes <= 0
            or self.concm_match_anchors is None
        ):
            return 0.0
        if self.concm_match_tail_weight:
            raise ValueError("concm_match_tail_weight is reserved for a later gate and must stay false in rung3.")

        epoch_one_based = epoch + 1
        if epoch_one_based <= self.concm_match_warmup_epoch:
            return 0.0
        return self.concm_match_lambda

    def _prepare_concm_match_anchors(self):
        self.concm_match_anchors = None
        self._last_match_anchor_diag = {}
        if (
            not self.concm_use_match_loss
            or self._cur_task <= 0
            or self._known_classes <= 0
        ):
            return
        if self.concm_match_tail_weight:
            raise ValueError("concm_match_tail_weight is reserved for a later gate and must stay false in rung3.")

        rng_state = self._capture_rng_state()
        try:
            self._network.eval()
            self._network.backbone.eval()
            self._network.original_backbone.eval()
            generator = torch.Generator()
            generator.manual_seed(int(self.args["seed"]) * 2000 + int(self._cur_task))
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                generator=generator,
            )
            cls_num_list = torch.Tensor(self.args["lt_list"][:self._total_classes]).to(self._device)

            current_main, current_few, current_labels = [], [], []
            with torch.no_grad():
                for _, inputs, targets in loader:
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    weight = cls_num_list[targets]
                    output = self._network(inputs, task_id=self._cur_task, weight=weight)
                    current_main.append(output["pre_logits"].detach().cpu())
                    current_few.append(output.get("pre_logits_few", output["pre_logits"]).detach().cpu())
                    current_labels.append(targets.detach().cpu())

            current_main = torch.cat(current_main, dim=0)
            current_few = torch.cat(current_few, dim=0)
            current_labels = torch.cat(current_labels, dim=0)

            dim_main = current_main.shape[1]
            dim_few = current_few.shape[1]
            anchors_main = torch.zeros(self._total_classes, dim_main)
            anchors_few = torch.zeros(self._total_classes, dim_few)
            valid_main = torch.zeros(self._total_classes, dtype=torch.bool)
            valid_few = torch.zeros(self._total_classes, dtype=torch.bool)

            old_count = 0
            for class_id, stats in self.concm_stage1_memory.items():
                class_id = int(class_id)
                if class_id >= self._known_classes:
                    continue
                anchors_main[class_id] = stats["mean_main"].detach().cpu()
                anchors_few[class_id] = stats["mean_few"].detach().cpu()
                valid_main[class_id] = True
                valid_few[class_id] = True
                old_count += 1

            current_classes = []
            for class_id in torch.unique(current_labels).tolist():
                class_id = int(class_id)
                mask = current_labels == class_id
                anchors_main[class_id] = current_main[mask].mean(dim=0)
                anchors_few[class_id] = current_few[mask].mean(dim=0)
                valid_main[class_id] = True
                valid_few[class_id] = True
                current_classes.append(class_id)

            finite_main = torch.isfinite(anchors_main[valid_main]).all().item() if valid_main.any() else True
            finite_few = torch.isfinite(anchors_few[valid_few]).all().item() if valid_few.any() else True
            nan_or_inf = int(not finite_main or not finite_few)
            if nan_or_inf:
                raise FloatingPointError("ConCM match anchors contain NaN or Inf.")

            self.concm_match_anchors = {
                "main": anchors_main,
                "few": anchors_few,
                "valid_main": valid_main,
                "valid_few": valid_few,
            }
            diag = {
                "concm_match_anchor_active": True,
                "concm_match_anchor_old_classes": int(old_count),
                "concm_match_anchor_current_classes": int(len(current_classes)),
                "concm_match_anchor_total_valid": int(valid_main.sum().item()),
                "concm_match_anchor_main_norm_mean": np.around(float(anchors_main[valid_main].norm(dim=1).mean().item()), decimals=6),
                "concm_match_anchor_few_norm_mean": np.around(float(anchors_few[valid_few].norm(dim=1).mean().item()), decimals=6),
                "concm_match_anchor_nan_or_inf": nan_or_inf,
            }
            self._last_match_anchor_diag = diag
            logging.info(
                (
                    "ConCMMatchAnchors task={} active=True old_classes={} current_classes={} "
                    "valid_classes={} main_norm_mean={:.6f} few_norm_mean={:.6f} nan_or_inf={}"
                ).format(
                    self._cur_task,
                    diag["concm_match_anchor_old_classes"],
                    diag["concm_match_anchor_current_classes"],
                    diag["concm_match_anchor_total_valid"],
                    diag["concm_match_anchor_main_norm_mean"],
                    diag["concm_match_anchor_few_norm_mean"],
                    diag["concm_match_anchor_nan_or_inf"],
                )
            )
        finally:
            self._restore_rng_state(rng_state)

    def _concm_feature_anchor_loss(self, output, targets):
        if self.concm_match_anchors is None:
            return None

        paths = self._concm_match_paths()
        losses = []
        details = {}
        for path in paths:
            feature_key = "pre_logits" if path == "main" else "pre_logits_few"
            if feature_key not in output:
                continue
            valid = self.concm_match_anchors["valid_{}".format(path)].to(self._device)
            mask = valid[targets]
            if self.concm_match_current_only:
                mask = mask & (targets >= self._known_classes) & (targets < self._total_classes)
            if not mask.any():
                continue

            features = output[feature_key][mask]
            anchors = self.concm_match_anchors[path].to(self._device)[targets[mask]]
            if self.concm_match_detach_anchors:
                anchors = anchors.detach()
            cosine = F.cosine_similarity(
                F.normalize(features, dim=1),
                F.normalize(anchors, dim=1),
                dim=1,
            )
            path_loss = (1.0 - cosine).mean()
            if not torch.isfinite(path_loss):
                raise FloatingPointError("ConCM match loss is NaN or Inf for path {}".format(path))
            losses.append(path_loss)
            details["{}_loss".format(path)] = path_loss
            details["{}_cosine".format(path)] = cosine.detach().mean()

        if len(losses) == 0:
            return None

        total = torch.stack(losses).mean()
        details["loss"] = total
        return details

    def _concm_usfm_paths(self):
        paths = str(self.concm_usfm_paths).lower().replace("+", ",").split(",")
        paths = [path.strip() for path in paths if path.strip()]
        if "both" in paths:
            return ["main", "few"]
        valid = []
        for path in paths:
            if path not in ("main", "few"):
                raise ValueError("Unsupported concm_usfm_paths: {}".format(self.concm_usfm_paths))
            valid.append(path)
        return valid

    def _reset_concm_usfm_anchors(self):
        self.concm_usfm_anchors = None
        self._last_usfm_anchor_diag = {}

    def _concm_usfm_effective_weight(self, epoch):
        if (
            not self.concm_use_uncertainty_selective_match
            or self.concm_usfm_lambda <= 0
            or self._cur_task <= 0
            or self._known_classes <= 0
        ):
            return 0.0
        epoch_one_based = epoch + 1
        if epoch_one_based <= self.concm_usfm_warmup_epoch:
            return 0.0
        return self.concm_usfm_lambda

    def _ensure_concm_usfm_anchors(self, output):
        if self.concm_usfm_anchors is not None:
            return
        main_dim = output["pre_logits"].shape[1]
        few_dim = output.get("pre_logits_few", output["pre_logits"]).shape[1]
        device = output["pre_logits"].device
        self.concm_usfm_anchors = {
            "main": torch.zeros(self._total_classes, main_dim, device=device),
            "few": torch.zeros(self._total_classes, few_dim, device=device),
            "valid_main": torch.zeros(self._total_classes, dtype=torch.bool, device=device),
            "valid_few": torch.zeros(self._total_classes, dtype=torch.bool, device=device),
        }

    def _update_concm_usfm_anchors(self, output, targets):
        if (
            not self.concm_use_uncertainty_selective_match
            or self._cur_task <= 0
            or self._known_classes <= 0
        ):
            return {}

        self._ensure_concm_usfm_anchors(output)
        current_mask = torch.logical_and(targets >= self._known_classes, targets < self._total_classes)
        if self.concm_usfm_current_real_only and not current_mask.any():
            return {}

        with torch.no_grad():
            paths = self._concm_usfm_paths()
            class_ids = targets[current_mask].detach().unique()
            for class_id in class_ids:
                class_id_int = int(class_id.item())
                mask = torch.logical_and(current_mask, targets == class_id)
                for path in paths:
                    feature_key = "pre_logits" if path == "main" else "pre_logits_few"
                    if feature_key not in output:
                        continue
                    feature_mean = output[feature_key].detach()[mask].mean(dim=0)
                    valid_key = "valid_{}".format(path)
                    if self.concm_usfm_anchors[valid_key][class_id_int]:
                        self.concm_usfm_anchors[path][class_id_int] = (
                            self.concm_usfm_anchor_momentum * self.concm_usfm_anchors[path][class_id_int]
                            + (1.0 - self.concm_usfm_anchor_momentum) * feature_mean
                        )
                    else:
                        self.concm_usfm_anchors[path][class_id_int] = feature_mean
                        self.concm_usfm_anchors[valid_key][class_id_int] = True

            valid_main = self.concm_usfm_anchors["valid_main"]
            valid_few = self.concm_usfm_anchors["valid_few"]
            finite_main = (
                torch.isfinite(self.concm_usfm_anchors["main"][valid_main]).all().item()
                if valid_main.any()
                else True
            )
            finite_few = (
                torch.isfinite(self.concm_usfm_anchors["few"][valid_few]).all().item()
                if valid_few.any()
                else True
            )
            nan_or_inf = int(not finite_main or not finite_few)
            if nan_or_inf:
                raise FloatingPointError("ConCM USFM anchors contain NaN or Inf.")

            diag = {
                "concm_usfm_anchor_current_classes": int(
                    torch.logical_and(
                        valid_main,
                        torch.arange(self._total_classes, device=valid_main.device) >= self._known_classes,
                    ).sum().item()
                ),
                "concm_usfm_anchor_main_norm_mean": 0.0 if not valid_main.any() else float(
                    self.concm_usfm_anchors["main"][valid_main].norm(dim=1).mean().item()
                ),
                "concm_usfm_anchor_few_norm_mean": 0.0 if not valid_few.any() else float(
                    self.concm_usfm_anchors["few"][valid_few].norm(dim=1).mean().item()
                ),
                "concm_usfm_anchor_nan_or_inf": nan_or_inf,
            }
            self._last_usfm_anchor_diag = diag
            return diag

    def _concm_usfm_gate(self, ale, epi, mask):
        ale_selected = ale.detach()[mask]
        epi_selected = epi.detach()[mask]
        if ale_selected.numel() == 0:
            return None

        ale_std = ale_selected.std(unbiased=False).clamp_min(1e-6)
        epi_std = epi_selected.std(unbiased=False).clamp_min(1e-6)
        ale_z = (ale_selected - ale_selected.mean()) / ale_std
        epi_z = (epi_selected - epi_selected.mean()) / epi_std
        gate = torch.sigmoid(self.concm_usfm_gate_k * (epi_z - ale_z)).clamp(0.0, 1.0)
        if self.concm_usfm_detach_gate:
            gate = gate.detach()
        return gate

    def _concm_usfm_loss(self, output, targets, ale, epi, all_seen_logits):
        if (
            not self.concm_use_uncertainty_selective_match
            or self.concm_usfm_anchors is None
            or ale is None
            or epi is None
            or all_seen_logits is None
            or self._cur_task <= 0
            or self._known_classes <= 0
        ):
            return None

        paths = self._concm_usfm_paths()
        current_mask = torch.logical_and(targets >= self._known_classes, targets < self._total_classes)
        if self.concm_usfm_current_real_only:
            base_mask = current_mask
        else:
            base_mask = torch.ones_like(targets, dtype=torch.bool)

        valid_mask = base_mask.clone()
        for path in paths:
            valid_key = "valid_{}".format(path)
            valid_mask = torch.logical_and(valid_mask, self.concm_usfm_anchors[valid_key][targets])
        if not valid_mask.any():
            return None

        gate = self._concm_usfm_gate(ale, epi, valid_mask)
        if gate is None:
            return None

        losses = []
        details = {}
        for path in paths:
            feature_key = "pre_logits" if path == "main" else "pre_logits_few"
            if feature_key not in output:
                continue
            features = output[feature_key][valid_mask]
            anchors = self.concm_usfm_anchors[path][targets[valid_mask]]
            if self.concm_usfm_detach_anchors:
                anchors = anchors.detach()
            cosine = F.cosine_similarity(
                F.normalize(features, dim=1),
                F.normalize(anchors, dim=1),
                dim=1,
            )
            path_loss = (gate * (1.0 - cosine)).mean()
            if not torch.isfinite(path_loss):
                raise FloatingPointError("ConCM USFM loss is NaN or Inf for path {}".format(path))
            losses.append(path_loss)
            details["{}_loss".format(path)] = path_loss
            details["{}_cosine".format(path)] = cosine.detach().mean()

        if len(losses) == 0:
            return None

        selected_targets = targets[valid_mask]
        selected_preds = all_seen_logits.detach()[valid_mask].argmax(dim=1)
        correct_mask = selected_preds == selected_targets
        new_pred_old_mask = torch.logical_and(selected_targets >= self._known_classes, selected_preds < self._known_classes)
        gate_detached = gate.detach()
        epi_detached = epi.detach()[valid_mask]
        ale_detached = ale.detach()[valid_mask]

        def _masked_gate_mean(mask):
            return None if not mask.any() else gate_detached[mask].mean()

        correct_gate = _masked_gate_mean(correct_mask)
        incorrect_gate = _masked_gate_mean(torch.logical_not(correct_mask))
        new_pred_old_gate = _masked_gate_mean(new_pred_old_mask)
        total = torch.stack(losses).sum()
        details.update(
            {
                "loss": total,
                "gate_mean": gate_detached.mean(),
                "gate_std": gate_detached.std(unbiased=False),
                "gate_min": gate_detached.min(),
                "gate_max": gate_detached.max(),
                "gate_correct_mean": correct_gate,
                "gate_incorrect_mean": incorrect_gate,
                "gate_new_pred_old_mean": new_pred_old_gate,
                "gate_correct_n": int(correct_mask.sum().item()),
                "gate_incorrect_n": int(torch.logical_not(correct_mask).sum().item()),
                "gate_new_pred_old_n": int(new_pred_old_mask.sum().item()),
                "ale_mean": ale_detached.mean(),
                "ale_std": ale_detached.std(unbiased=False),
                "epi_mean": epi_detached.mean(),
                "epi_std": epi_detached.std(unbiased=False),
                "selected_n": int(valid_mask.sum().item()),
            }
        )
        return details

    def _capture_rng_state(self):
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])

    def _concm_proto_group(self, count):
        if count > 100:
            return "many"
        if count > 20:
            return "medium"
        return "few"

    def _concm_proto_candidate_bank(self, raw_stats, path):
        mean_key = "mean_{}".format(path)
        var_key = "var_{}".format(path)
        raw_mean_key = "raw_mean_{}".format(path)
        raw_var_key = "raw_var_{}".format(path)
        bank = []

        for class_id, stats in self.concm_stage1_memory.items():
            bank.append(
                {
                    "class_id": int(class_id),
                    "mean": stats.get(raw_mean_key, stats[mean_key]).detach().cpu(),
                    "var": stats.get(raw_var_key, stats[var_key]).detach().cpu(),
                    "count": int(stats["count"]),
                }
            )

        for class_id, stats in raw_stats.items():
            bank.append(
                {
                    "class_id": int(class_id),
                    "mean": stats[mean_key].detach().cpu(),
                    "var": stats[var_key].detach().cpu(),
                    "count": int(stats["count"]),
                }
            )

        return bank

    def _concm_proto_diag_summary(self, path, rows):
        prefix = "concm_proto_{}".format(path)
        summary = {}
        if len(rows) == 0:
            summary["{}_classes".format(prefix)] = 0
            summary["{}_nan_or_inf".format(prefix)] = 0
            return summary

        summary["{}_classes".format(prefix)] = len(rows)
        summary["{}_nan_or_inf".format(prefix)] = int(sum(row["nan_or_inf"] for row in rows))

        for field in ["alpha", "raw_calib_cos"]:
            values = [row[field] for row in rows if np.isfinite(row[field])]
            key = "{}_mean".format(field)
            summary["{}_{}".format(prefix, key)] = 0.0 if len(values) == 0 else np.around(float(np.mean(values)), decimals=6)
            for group in ["many", "medium", "few"]:
                group_values = [row[field] for row in rows if row["group"] == group and np.isfinite(row[field])]
                summary["{}_{}_{}_mean".format(prefix, field, group)] = (
                    0.0 if len(group_values) == 0 else np.around(float(np.mean(group_values)), decimals=6)
                )
                summary["{}_{}_classes".format(prefix, group)] = int(sum(row["group"] == group for row in rows))

        return summary

    def _calibrate_concm_stage1_path(self, raw_stats, path):
        mean_key = "mean_{}".format(path)
        var_key = "var_{}".format(path)
        bank = self._concm_proto_candidate_bank(raw_stats, path)
        calibrated = {}
        rows = []

        if len(bank) == 0:
            return calibrated, self._concm_proto_diag_summary(path, rows)

        mean_vars = [float(item["var"].float().mean().item()) for item in bank]
        max_mean_var = max(max(mean_vars), 1e-12)
        log_counts = torch.tensor([np.log(float(item["count"]) + 1.0) for item in bank], dtype=torch.float32)
        median_log_count = log_counts.median()
        max_log_count = log_counts.max().clamp_min(1e-12)

        for class_id, stats in raw_stats.items():
            target_mean = stats[mean_key].float().detach().cpu()
            target_var = stats[var_key].float().detach().cpu()
            target_count = int(stats["count"])
            group = self._concm_proto_group(target_count)
            target_norm = target_mean.norm().clamp_min(1e-12)
            target_dir = F.normalize(target_mean.unsqueeze(0), dim=1).squeeze(0)
            target_uncertainty = torch.tensor(float(target_var.mean().item()) / max_mean_var, dtype=torch.float32).clamp(0.0, 1.0)
            target_log_count = torch.tensor(np.log(float(target_count) + 1.0), dtype=torch.float32)
            alpha = torch.sigmoid(self.concm_alpha_a * (target_log_count - median_log_count) - self.concm_alpha_b * target_uncertainty)
            alpha = alpha.clamp(self.concm_alpha_min, self.concm_alpha_max)

            candidates = [item for item in bank if int(item["class_id"]) != int(class_id)]
            nan_or_inf = 0
            if len(candidates) == 0 or target_mean.numel() == 0:
                calibrated_mean = target_mean
                calibrated_var = target_var.clamp_min(0.0)
                raw_calib_cos = 1.0
                alpha_value = 1.0
            else:
                candidate_means = torch.stack([item["mean"].float() for item in candidates], dim=0)
                candidate_vars = torch.stack([item["var"].float() for item in candidates], dim=0)
                candidate_counts = torch.tensor([float(item["count"]) for item in candidates], dtype=torch.float32)
                candidate_dirs = F.normalize(candidate_means, dim=1)
                candidate_uncertainty = (candidate_vars.mean(dim=1) / max_mean_var).clamp(0.0, 1.0)
                candidate_reliability = (torch.log(candidate_counts + 1.0) / max_log_count).clamp(0.0, 1.0)
                candidate_reliability = (candidate_reliability * (1.0 - candidate_uncertainty)).clamp_min(1e-6)
                scores = torch.matmul(candidate_dirs, target_dir) * candidate_reliability
                topk = min(max(self.concm_proto_topk, 1), scores.numel())
                top_scores, top_indices = torch.topk(scores, k=topk)
                weights = torch.softmax(top_scores, dim=0)
                neighbor_dir = torch.sum(candidate_dirs[top_indices] * weights.unsqueeze(1), dim=0)
                neighbor_norm = neighbor_dir.norm()
                if not torch.isfinite(neighbor_norm) or neighbor_norm.item() <= 1e-12:
                    calibrated_mean = target_mean
                    calibrated_var = target_var.clamp_min(0.0)
                    raw_calib_cos = 1.0
                    alpha_value = 1.0
                else:
                    neighbor_dir = neighbor_dir / neighbor_norm.clamp_min(1e-12)
                    calibrated_dir = alpha * target_dir + (1.0 - alpha) * neighbor_dir
                    calibrated_dir = F.normalize(calibrated_dir.unsqueeze(0), dim=1).squeeze(0)
                    calibrated_mean = calibrated_dir * target_norm
                    neighbor_var = torch.sum(candidate_vars[top_indices] * weights.unsqueeze(1), dim=0)
                    calibrated_var = (alpha * target_var + (1.0 - alpha) * neighbor_var).clamp_min(0.0)
                    raw_calib_cos = F.cosine_similarity(target_mean.unsqueeze(0), calibrated_mean.unsqueeze(0), dim=1).item()
                    alpha_value = float(alpha.item())

            if not torch.isfinite(calibrated_mean).all() or not torch.isfinite(calibrated_var).all():
                nan_or_inf = 1
                calibrated_mean = target_mean
                calibrated_var = target_var.clamp_min(0.0)
                raw_calib_cos = 1.0
                alpha_value = 1.0

            calibrated[int(class_id)] = {
                mean_key: calibrated_mean,
                var_key: calibrated_var,
                "alpha_{}".format(path): alpha_value,
                "raw_calib_cos_{}".format(path): float(raw_calib_cos),
            }
            rows.append(
                {
                    "class_id": int(class_id),
                    "count": target_count,
                    "group": group,
                    "alpha": alpha_value,
                    "raw_calib_cos": float(raw_calib_cos),
                    "nan_or_inf": nan_or_inf,
                }
            )

        return calibrated, self._concm_proto_diag_summary(path, rows)

    def _apply_concm_proto_calibration(self, raw_stats):
        main_calibrated, main_diag = self._calibrate_concm_stage1_path(raw_stats, "main")
        few_calibrated, few_diag = self._calibrate_concm_stage1_path(raw_stats, "few")
        stats_to_store = {}
        for class_id, stats in raw_stats.items():
            class_id = int(class_id)
            stats_to_store[class_id] = {
                "mean_main": main_calibrated[class_id]["mean_main"],
                "var_main": main_calibrated[class_id]["var_main"],
                "mean_few": few_calibrated[class_id]["mean_few"],
                "var_few": few_calibrated[class_id]["var_few"],
                "raw_mean_main": stats["mean_main"],
                "raw_var_main": stats["var_main"],
                "raw_mean_few": stats["mean_few"],
                "raw_var_few": stats["var_few"],
                "count": int(stats["count"]),
                "task": int(stats["task"]),
                "concm_proto_alpha_main": main_calibrated[class_id]["alpha_main"],
                "concm_proto_alpha_few": few_calibrated[class_id]["alpha_few"],
                "concm_proto_raw_calib_cos_main": main_calibrated[class_id]["raw_calib_cos_main"],
                "concm_proto_raw_calib_cos_few": few_calibrated[class_id]["raw_calib_cos_few"],
            }

        diag = {"concm_proto_calibration_active": True}
        diag.update(main_diag)
        diag.update(few_diag)
        diag["concm_proto_calibration_nan_or_inf"] = int(
            diag.get("concm_proto_main_nan_or_inf", 0) + diag.get("concm_proto_few_nan_or_inf", 0)
        )
        self._last_proto_calibration_diag = diag
        return stats_to_store, diag

    def _update_concm_stage1_memory(self):
        rng_state = self._capture_rng_state()
        try:
            self._network.eval()
            self._network.backbone.eval()
            self._network.original_backbone.eval()
            generator = torch.Generator()
            generator.manual_seed(int(self.args["seed"]) * 1000 + int(self._cur_task))
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                generator=generator,
            )
            cls_num_list = torch.Tensor(self.args["lt_list"][:self._total_classes]).to(self._device)

            main_features, few_features, labels = [], [], []
            with torch.no_grad():
                for _, inputs, targets in loader:
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    weight = cls_num_list[targets]
                    output = self._network(inputs, task_id=self._cur_task, weight=weight)
                    main_features.append(output["pre_logits"].detach().cpu())
                    few_features.append(output.get("pre_logits_few", output["pre_logits"]).detach().cpu())
                    labels.append(targets.detach().cpu())

            main_features = torch.cat(main_features, dim=0)
            few_features = torch.cat(few_features, dim=0)
            labels = torch.cat(labels, dim=0)

            raw_stats = {}
            updated = []
            for class_id in torch.unique(labels).tolist():
                mask = labels == class_id
                class_main = main_features[mask]
                class_few = few_features[mask]
                raw_stats[int(class_id)] = {
                    "mean_main": class_main.mean(dim=0),
                    "var_main": class_main.var(dim=0, unbiased=False),
                    "mean_few": class_few.mean(dim=0),
                    "var_few": class_few.var(dim=0, unbiased=False),
                    "count": int(mask.sum().item()),
                    "task": int(self._cur_task),
                }
                updated.append(int(class_id))

            if self.concm_proto_calibration:
                stats_to_store, proto_diag = self._apply_concm_proto_calibration(raw_stats)
            else:
                stats_to_store = raw_stats
                proto_diag = {}
                self._last_proto_calibration_diag = {}

            for class_id in updated:
                self.concm_stage1_memory[int(class_id)] = stats_to_store[int(class_id)]
        finally:
            self._restore_rng_state(rng_state)

        logging.info(
            "ConCM-lite Stage1 memory updated: task={}, classes={}, total_memory_classes={}, synth_per_class={}, loss_weight={}".format(
                self._cur_task,
                updated,
                len(self.concm_stage1_memory),
                self.concm_stage1_synth_per_class,
                self.concm_stage1_loss_weight,
            )
        )
        if self.concm_proto_calibration:
            logging.info(
                (
                    "ConCMProtoCalibration task={} active=True classes={} main_alpha_mean={:.6f} "
                    "few_alpha_mean={:.6f} main_raw_calib_cos_mean={:.6f} few_raw_calib_cos_mean={:.6f} "
                    "nan_or_inf={}"
                ).format(
                    self._cur_task,
                    len(updated),
                    proto_diag.get("concm_proto_main_alpha_mean", 0.0),
                    proto_diag.get("concm_proto_few_alpha_mean", 0.0),
                    proto_diag.get("concm_proto_main_raw_calib_cos_mean", 0.0),
                    proto_diag.get("concm_proto_few_raw_calib_cos_mean", 0.0),
                    proto_diag.get("concm_proto_calibration_nan_or_inf", 0),
                )
            )

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        cls_num_list = torch.Tensor(self.args["lt_list"][:self._total_classes]).to(self._device)
        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            self._network.original_backbone.eval()
            losses = 0.0
            ce_losses = 0.0
            concm_stage1_raw_losses = 0.0
            concm_stage1_weighted_losses = 0.0
            concm_stage1_raw_ratios = 0.0
            concm_stage1_weighted_ratios = 0.0
            concm_stage1_effective_weights = 0.0
            concm_stage1_batches = 0
            concm_stage1_nonzero_grad_batches = 0
            concm_replay_old_rates = 0.0
            concm_replay_scales = 0.0
            concm_replay_effective_caps = 0.0
            concm_replay_controller_batches = 0
            concm_ugr_batch_ales = 0.0
            concm_ugr_batch_epis = 0.0
            concm_ugr_old_rates = 0.0
            concm_ugr_scales = 0.0
            concm_ugr_effective_caps = 0.0
            concm_ugr_batches = 0
            concm_match_raw_losses = 0.0
            concm_match_weighted_losses = 0.0
            concm_match_main_losses = 0.0
            concm_match_few_losses = 0.0
            concm_match_main_cosines = 0.0
            concm_match_few_cosines = 0.0
            concm_match_weighted_ratios = 0.0
            concm_match_raw_ratios = 0.0
            concm_match_effective_weights = 0.0
            concm_match_batches = 0
            concm_match_nonzero_grad_batches = 0
            concm_usfm_raw_losses = 0.0
            concm_usfm_weighted_losses = 0.0
            concm_usfm_main_losses = 0.0
            concm_usfm_few_losses = 0.0
            concm_usfm_main_cosines = 0.0
            concm_usfm_few_cosines = 0.0
            concm_usfm_weighted_ratios = 0.0
            concm_usfm_effective_weights = 0.0
            concm_usfm_gate_means = 0.0
            concm_usfm_gate_stds = 0.0
            concm_usfm_gate_min = None
            concm_usfm_gate_max = None
            concm_usfm_gate_correct_means = 0.0
            concm_usfm_gate_correct_batches = 0
            concm_usfm_gate_incorrect_means = 0.0
            concm_usfm_gate_incorrect_batches = 0
            concm_usfm_gate_new_pred_old_means = 0.0
            concm_usfm_gate_new_pred_old_batches = 0
            concm_usfm_gate_correct_ns = 0.0
            concm_usfm_gate_incorrect_ns = 0.0
            concm_usfm_gate_new_pred_old_ns = 0.0
            concm_usfm_ale_means = 0.0
            concm_usfm_ale_stds = 0.0
            concm_usfm_epi_means = 0.0
            concm_usfm_epi_stds = 0.0
            concm_usfm_selected_ns = 0.0
            concm_usfm_batches = 0
            concm_usfm_nonzero_grad_batches = 0
            unc_train_ale = 0.0
            unc_train_epi = 0.0
            unc_train_error_rates = 0.0
            unc_train_current_pred_old_rates = 0.0
            unc_train_batches = 0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                weight = cls_num_list[targets]

                output = self._network(inputs, task_id=self._cur_task, train=True, weight=weight) 
                unc = None
                ale = None
                epi = None
                all_seen_unc_logits = None
                if (
                    self.concm_uncertainty_diagnostics
                    or self.concm_uncertainty_gated_replay
                    or self.concm_use_uncertainty_selective_match
                ):
                    logits_main_all_seen = output["logits"][:, : self._total_classes]
                    logits_few_all_seen = output["logits_few"][:, : self._total_classes]
                    all_seen_unc_logits = logits_main_all_seen + logits_few_all_seen
                    unc = self._concm_uncertainty_from_logits(logits_main_all_seen, logits_few_all_seen)
                    if self.concm_uncertainty_diagnostics and unc is not None:
                        ale, epi = unc
                        self._update_uncertainty_ema(ale, epi, targets)
                        with torch.no_grad():
                            unc_preds = all_seen_unc_logits.detach().argmax(dim=1)
                            unc_train_ale += ale.mean().item()
                            unc_train_epi += epi.mean().item()
                            unc_train_error_rates += (unc_preds != targets).float().mean().item() * 100.0
                            if self._known_classes > 0:
                                unc_train_current_pred_old_rates += (unc_preds < self._known_classes).float().mean().item() * 100.0
                            unc_train_batches += 1
                    elif unc is not None:
                        ale, epi = unc

                pool = output["pool_id"]
                logits = output["logits"]
                logits = logits[:, self._known_classes : self._total_classes] 
                fake_targets = targets - self._known_classes
                loss1 = F.cross_entropy(logits, fake_targets.long())
                loss = loss1

                
                logits_few = output["logits_few"]
                logits_few = logits_few[:, self._known_classes : self._total_classes] 
                logits_all = logits + logits_few
                target = F.one_hot(fake_targets, self._total_classes - self._known_classes)
                
                loss_all = F.cross_entropy(logits_all, fake_targets.long())
                loss += loss_all
                ce_losses += loss_all.item()
                loss_few = - (pool.squeeze(1) * (target * torch.log(nn.Softmax(dim=-1)(logits_few)+1e-7)).sum(dim=1)).sum()
                loss += loss_few
                loss /= 3  
                    
                weight_norm = torch.where(weight <= self.theta, 1.0, 0.1) 
                weight_weight = torch.where(weight <= self.theta, 10.0, 1.0)
                    
                if epoch < 5:
                    match_loss = ((weight_norm - pool.squeeze(1))**2 * weight_weight).sum()
                else:
                    match_loss  = ((1.0 - pool.squeeze(1)) ** 2 * pool.squeeze(1) * 10).sum()

                loss += match_loss

                if self.args["pull_constraint"] and 'reduce_sim' in output:  
                    loss = loss - self.args["pull_constraint_coeff"] * output['reduce_sim'] 
                    loss -= self.args["pull_constraint_coeff"] * output['reduce_sim_few']

                effective_stage1_weight = self._concm_stage1_effective_weight(epoch)
                stage1_max_synth_total = None
                controller_state = None
                if self.concm_replay_bias_controller and effective_stage1_weight > 0:
                    all_seen_train_logits = (
                        output["logits"][:, : self._total_classes]
                        + output["logits_few"][:, : self._total_classes]
                    )
                    controller_state = self._concm_replay_controller_state(all_seen_train_logits)
                    if controller_state is not None:
                        effective_stage1_weight *= controller_state["replay_scale"]
                        stage1_max_synth_total = controller_state["effective_synth_cap"]

                ugr_state = None
                if self.concm_uncertainty_gated_replay and effective_stage1_weight > 0:
                    ugr_state = self._concm_uncertainty_gated_replay_state(
                        ale,
                        epi,
                        all_seen_unc_logits,
                        targets,
                    )
                    if ugr_state is not None:
                        effective_stage1_weight *= ugr_state["replay_scale"]
                        stage1_max_synth_total = ugr_state["effective_synth_cap"]

                concm_stage1_loss = self._concm_stage1_loss(max_synth_total=stage1_max_synth_total)
                if concm_stage1_loss is not None:
                    weighted_concm_stage1_loss = effective_stage1_weight * concm_stage1_loss
                    if effective_stage1_weight > 0:
                        backbone = self._backbone_module()
                        grad_params = list(backbone.head.parameters()) + list(backbone.head_few.parameters())
                        stage1_grads = torch.autograd.grad(
                            weighted_concm_stage1_loss,
                            grad_params,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        if any(
                            grad is not None
                            and torch.isfinite(grad).all()
                            and grad.detach().abs().max().item() > 0
                            for grad in stage1_grads
                        ):
                            concm_stage1_nonzero_grad_batches += 1

                        loss += weighted_concm_stage1_loss
                    raw_stage1_value = concm_stage1_loss.item()
                    weighted_stage1_value = weighted_concm_stage1_loss.item()
                    ce_value = loss_all.item()
                    concm_stage1_raw_losses += raw_stage1_value
                    concm_stage1_weighted_losses += weighted_stage1_value
                    concm_stage1_raw_ratios += raw_stage1_value / (ce_value + 1e-12)
                    concm_stage1_weighted_ratios += weighted_stage1_value / (ce_value + 1e-12)
                    concm_stage1_effective_weights += effective_stage1_weight
                    concm_stage1_batches += 1
                    if controller_state is not None:
                        concm_replay_old_rates += controller_state["train_current_pred_old_rate"]
                        concm_replay_scales += controller_state["replay_scale"]
                        concm_replay_effective_caps += controller_state["effective_synth_cap"]
                        concm_replay_controller_batches += 1
                    if ugr_state is not None:
                        concm_ugr_batch_ales += ugr_state["batch_Ale_current"]
                        concm_ugr_batch_epis += ugr_state["batch_Epi_current"]
                        concm_ugr_old_rates += ugr_state["train_current_pred_old_rate"]
                        concm_ugr_scales += ugr_state["replay_scale"]
                        concm_ugr_effective_caps += ugr_state["effective_synth_cap"]
                        concm_ugr_batches += 1

                if self.concm_use_uncertainty_selective_match:
                    self._update_concm_usfm_anchors(output, targets)

                effective_usfm_weight = self._concm_usfm_effective_weight(epoch)
                if effective_usfm_weight > 0:
                    concm_usfm = self._concm_usfm_loss(output, targets, ale, epi, all_seen_unc_logits)
                    if concm_usfm is not None:
                        raw_usfm_loss = concm_usfm["loss"]
                        weighted_usfm_loss = effective_usfm_weight * raw_usfm_loss
                        backbone = self._backbone_module()
                        usfm_grad_params = [
                            p
                            for name, p in backbone.named_parameters()
                            if p.requires_grad
                            and not name.startswith("head.")
                            and not name.startswith("head_few.")
                        ]
                        usfm_grads = torch.autograd.grad(
                            weighted_usfm_loss,
                            usfm_grad_params,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        if any(
                            grad is not None
                            and torch.isfinite(grad).all()
                            and grad.detach().abs().max().item() > 0
                            for grad in usfm_grads
                        ):
                            concm_usfm_nonzero_grad_batches += 1

                        loss += weighted_usfm_loss
                        raw_usfm_value = raw_usfm_loss.item()
                        weighted_usfm_value = weighted_usfm_loss.item()
                        ce_value = loss_all.item()
                        concm_usfm_raw_losses += raw_usfm_value
                        concm_usfm_weighted_losses += weighted_usfm_value
                        concm_usfm_weighted_ratios += weighted_usfm_value / (ce_value + 1e-12)
                        concm_usfm_effective_weights += effective_usfm_weight
                        concm_usfm_gate_means += concm_usfm["gate_mean"].item()
                        concm_usfm_gate_stds += concm_usfm["gate_std"].item()
                        gate_min_value = concm_usfm["gate_min"].item()
                        gate_max_value = concm_usfm["gate_max"].item()
                        concm_usfm_gate_min = (
                            gate_min_value
                            if concm_usfm_gate_min is None
                            else min(concm_usfm_gate_min, gate_min_value)
                        )
                        concm_usfm_gate_max = (
                            gate_max_value
                            if concm_usfm_gate_max is None
                            else max(concm_usfm_gate_max, gate_max_value)
                        )
                        if concm_usfm["gate_correct_mean"] is not None:
                            concm_usfm_gate_correct_means += concm_usfm["gate_correct_mean"].item()
                            concm_usfm_gate_correct_batches += 1
                        if concm_usfm["gate_incorrect_mean"] is not None:
                            concm_usfm_gate_incorrect_means += concm_usfm["gate_incorrect_mean"].item()
                            concm_usfm_gate_incorrect_batches += 1
                        if concm_usfm["gate_new_pred_old_mean"] is not None:
                            concm_usfm_gate_new_pred_old_means += concm_usfm["gate_new_pred_old_mean"].item()
                            concm_usfm_gate_new_pred_old_batches += 1
                        concm_usfm_gate_correct_ns += concm_usfm["gate_correct_n"]
                        concm_usfm_gate_incorrect_ns += concm_usfm["gate_incorrect_n"]
                        concm_usfm_gate_new_pred_old_ns += concm_usfm["gate_new_pred_old_n"]
                        concm_usfm_ale_means += concm_usfm["ale_mean"].item()
                        concm_usfm_ale_stds += concm_usfm["ale_std"].item()
                        concm_usfm_epi_means += concm_usfm["epi_mean"].item()
                        concm_usfm_epi_stds += concm_usfm["epi_std"].item()
                        concm_usfm_selected_ns += concm_usfm["selected_n"]
                        concm_usfm_batches += 1
                        if "main_loss" in concm_usfm:
                            concm_usfm_main_losses += concm_usfm["main_loss"].item()
                            concm_usfm_main_cosines += concm_usfm["main_cosine"].item()
                        if "few_loss" in concm_usfm:
                            concm_usfm_few_losses += concm_usfm["few_loss"].item()
                            concm_usfm_few_cosines += concm_usfm["few_cosine"].item()

                effective_match_weight = self._concm_match_effective_weight(epoch)
                if effective_match_weight > 0:
                    concm_match = self._concm_feature_anchor_loss(output, targets)
                    if concm_match is not None:
                        raw_match_loss = concm_match["loss"]
                        weighted_match_loss = effective_match_weight * raw_match_loss
                        backbone = self._backbone_module()
                        match_grad_params = [
                            p
                            for name, p in backbone.named_parameters()
                            if p.requires_grad
                            and not name.startswith("head.")
                            and not name.startswith("head_few.")
                        ]
                        match_grads = torch.autograd.grad(
                            weighted_match_loss,
                            match_grad_params,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        if any(
                            grad is not None
                            and torch.isfinite(grad).all()
                            and grad.detach().abs().max().item() > 0
                            for grad in match_grads
                        ):
                            concm_match_nonzero_grad_batches += 1

                        loss += weighted_match_loss
                        raw_match_value = raw_match_loss.item()
                        weighted_match_value = weighted_match_loss.item()
                        ce_value = loss_all.item()
                        concm_match_raw_losses += raw_match_value
                        concm_match_weighted_losses += weighted_match_value
                        concm_match_weighted_ratios += weighted_match_value / (ce_value + 1e-12)
                        concm_match_raw_ratios += raw_match_value / (ce_value + 1e-12)
                        concm_match_effective_weights += effective_match_weight
                        concm_match_batches += 1
                        if "main_loss" in concm_match:
                            concm_match_main_losses += concm_match["main_loss"].item()
                            concm_match_main_cosines += concm_match["main_cosine"].item()
                        if "few_loss" in concm_match:
                            concm_match_few_losses += concm_match["few_loss"].item()
                            concm_match_few_cosines += concm_match["few_cosine"].item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits_all, dim=1)

                correct += preds.eq(fake_targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            
            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy(pool1 {:.2f}, pool2 {:.2f}, pool_all {:.2f})".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc[0],test_acc[1], test_acc[2]
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                )
            if self.concm_stage1_enabled and concm_stage1_batches > 0:
                info += (
                    ", ConCMStage1_loss_raw {:.6f}, "
                    "ConCMStage1_loss_weighted {:.6f}, "
                    "ConCMStage1_effective_weight {:.6f}, "
                    "CE_loss {:.6f}, "
                    "ConCMStage1_to_CE {:.6f}, "
                    "ConCMStage1_raw_to_CE {:.6f}, "
                    "ConCMStage1_nonzero_grad_batches {}/{}"
                ).format(
                    concm_stage1_raw_losses / concm_stage1_batches,
                    concm_stage1_weighted_losses / concm_stage1_batches,
                    concm_stage1_effective_weights / concm_stage1_batches,
                    ce_losses / len(train_loader),
                    concm_stage1_weighted_ratios / concm_stage1_batches,
                    concm_stage1_raw_ratios / concm_stage1_batches,
                    concm_stage1_nonzero_grad_batches,
                    concm_stage1_batches,
                )
            if self.concm_replay_bias_controller and concm_replay_controller_batches > 0:
                info += (
                    ", ConCMReplay_train_current_pred_old_rate {:.2f}, "
                    "ConCMReplay_scale {:.6f}, "
                    "ConCMReplay_effective_stage1_loss_weight {:.6f}, "
                    "ConCMReplay_effective_synth_cap {:.2f}"
                ).format(
                    concm_replay_old_rates / concm_replay_controller_batches,
                    concm_replay_scales / concm_replay_controller_batches,
                    concm_stage1_effective_weights / concm_stage1_batches,
                    concm_replay_effective_caps / concm_replay_controller_batches,
                )
            if self.concm_uncertainty_gated_replay and concm_ugr_batches > 0:
                info += (
                    ", ConCMUGR_batch_Ale_current {:.6f}, "
                    "ConCMUGR_batch_Epi_current {:.6f}, "
                    "ConCMUGR_train_current_pred_old_rate {:.2f}, "
                    "ConCMUGR_replay_scale {:.6f}, "
                    "ConCMUGR_effective_stage1_loss_weight {:.6f}, "
                    "ConCMUGR_effective_synth_cap {:.2f}"
                ).format(
                    concm_ugr_batch_ales / concm_ugr_batches,
                    concm_ugr_batch_epis / concm_ugr_batches,
                    concm_ugr_old_rates / concm_ugr_batches,
                    concm_ugr_scales / concm_ugr_batches,
                    concm_stage1_effective_weights / concm_stage1_batches,
                    concm_ugr_effective_caps / concm_ugr_batches,
                )
            if self.concm_uncertainty_diagnostics and unc_train_batches > 0:
                current_pred_old_rate = (
                    unc_train_current_pred_old_rates / unc_train_batches
                    if self._known_classes > 0
                    else 0.0
                )
                info += (
                    ", ConCMUncertaintyTrain_Ale {:.6f}, "
                    "ConCMUncertaintyTrain_Epi {:.6f}, "
                    "ConCMUncertaintyTrain_error_rate {:.2f}, "
                    "ConCMUncertaintyTrain_current_pred_old_rate {:.2f}"
                ).format(
                    unc_train_ale / unc_train_batches,
                    unc_train_epi / unc_train_batches,
                    unc_train_error_rates / unc_train_batches,
                    current_pred_old_rate,
                )
            if self.concm_use_uncertainty_selective_match and concm_usfm_batches > 0:
                usfm_anchor_diag = self._last_usfm_anchor_diag or {}
                gate_correct_mean = (
                    -1.0
                    if concm_usfm_gate_correct_batches == 0
                    else concm_usfm_gate_correct_means / concm_usfm_gate_correct_batches
                )
                gate_incorrect_mean = (
                    -1.0
                    if concm_usfm_gate_incorrect_batches == 0
                    else concm_usfm_gate_incorrect_means / concm_usfm_gate_incorrect_batches
                )
                gate_new_pred_old_mean = (
                    -1.0
                    if concm_usfm_gate_new_pred_old_batches == 0
                    else concm_usfm_gate_new_pred_old_means / concm_usfm_gate_new_pred_old_batches
                )
                info += (
                    ", ConCMUSFM_loss_raw {:.6f}, "
                    "ConCMUSFM_loss_weighted {:.6f}, "
                    "ConCMUSFM_main_loss {:.6f}, "
                    "ConCMUSFM_few_loss {:.6f}, "
                    "ConCMUSFM_main_cos {:.6f}, "
                    "ConCMUSFM_few_cos {:.6f}, "
                    "ConCMUSFM_effective_weight {:.6f}, "
                    "ConCMUSFM_to_CE {:.6f}, "
                    "ConCMUSFM_Ale_mean {:.6f}, "
                    "ConCMUSFM_Ale_std {:.6f}, "
                    "ConCMUSFM_Epi_mean {:.6f}, "
                    "ConCMUSFM_Epi_std {:.6f}, "
                    "ConCMUSFM_gate_mean {:.6f}, "
                    "ConCMUSFM_gate_std {:.6f}, "
                    "ConCMUSFM_gate_min {:.6f}, "
                    "ConCMUSFM_gate_max {:.6f}, "
                    "ConCMUSFM_gate_correct {:.6f}, "
                    "ConCMUSFM_gate_incorrect {:.6f}, "
                    "ConCMUSFM_gate_new_pred_old {:.6f}, "
                    "ConCMUSFM_gate_correct_n {:.2f}, "
                    "ConCMUSFM_gate_incorrect_n {:.2f}, "
                    "ConCMUSFM_gate_new_pred_old_n {:.2f}, "
                    "ConCMUSFM_selected_n {:.2f}, "
                    "ConCMUSFM_anchor_classes {}, "
                    "ConCMUSFM_anchor_main_norm {:.6f}, "
                    "ConCMUSFM_anchor_few_norm {:.6f}, "
                    "ConCMUSFM_anchor_nan_or_inf {}, "
                    "ConCMUSFM_nonzero_grad_batches {}/{}"
                ).format(
                    concm_usfm_raw_losses / concm_usfm_batches,
                    concm_usfm_weighted_losses / concm_usfm_batches,
                    concm_usfm_main_losses / concm_usfm_batches,
                    concm_usfm_few_losses / concm_usfm_batches,
                    concm_usfm_main_cosines / concm_usfm_batches,
                    concm_usfm_few_cosines / concm_usfm_batches,
                    concm_usfm_effective_weights / concm_usfm_batches,
                    concm_usfm_weighted_ratios / concm_usfm_batches,
                    concm_usfm_ale_means / concm_usfm_batches,
                    concm_usfm_ale_stds / concm_usfm_batches,
                    concm_usfm_epi_means / concm_usfm_batches,
                    concm_usfm_epi_stds / concm_usfm_batches,
                    concm_usfm_gate_means / concm_usfm_batches,
                    concm_usfm_gate_stds / concm_usfm_batches,
                    0.0 if concm_usfm_gate_min is None else concm_usfm_gate_min,
                    0.0 if concm_usfm_gate_max is None else concm_usfm_gate_max,
                    gate_correct_mean,
                    gate_incorrect_mean,
                    gate_new_pred_old_mean,
                    concm_usfm_gate_correct_ns / concm_usfm_batches,
                    concm_usfm_gate_incorrect_ns / concm_usfm_batches,
                    concm_usfm_gate_new_pred_old_ns / concm_usfm_batches,
                    concm_usfm_selected_ns / concm_usfm_batches,
                    usfm_anchor_diag.get("concm_usfm_anchor_current_classes", 0),
                    usfm_anchor_diag.get("concm_usfm_anchor_main_norm_mean", 0.0),
                    usfm_anchor_diag.get("concm_usfm_anchor_few_norm_mean", 0.0),
                    usfm_anchor_diag.get("concm_usfm_anchor_nan_or_inf", 0),
                    concm_usfm_nonzero_grad_batches,
                    concm_usfm_batches,
                )
            if self.concm_use_match_loss and concm_match_batches > 0:
                info += (
                    ", ConCMMatch_loss_raw {:.6f}, "
                    "ConCMMatch_loss_weighted {:.6f}, "
                    "ConCMMatch_main_loss {:.6f}, "
                    "ConCMMatch_few_loss {:.6f}, "
                    "ConCMMatch_main_cos {:.6f}, "
                    "ConCMMatch_few_cos {:.6f}, "
                    "ConCMMatch_effective_weight {:.6f}, "
                    "ConCMMatch_to_CE {:.6f}, "
                    "ConCMMatch_raw_to_CE {:.6f}, "
                    "ConCMMatch_nonzero_grad_batches {}/{}"
                ).format(
                    concm_match_raw_losses / concm_match_batches,
                    concm_match_weighted_losses / concm_match_batches,
                    concm_match_main_losses / concm_match_batches,
                    concm_match_few_losses / concm_match_batches,
                    concm_match_main_cosines / concm_match_batches,
                    concm_match_few_cosines / concm_match_batches,
                    concm_match_effective_weights / concm_match_batches,
                    concm_match_weighted_ratios / concm_match_batches,
                    concm_match_raw_ratios / concm_match_batches,
                    concm_match_nonzero_grad_batches,
                    concm_match_batches,
                )
            prog_bar.set_description(info)
            logging.info(info)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true, y_logits = [], [], []
        y_unc_ale, y_unc_epi, y_uncalibrated_pred = [], [], []
        calibration_info = self._eval_calibration_info()
        self._last_eval_calibration = None
        self._last_uncertainty_diag = {}
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                res = self._network(inputs, task_id=self._cur_task)
                logits_main = res["logits"][:, :self._total_classes]
                logits_few = res["logits_few"][:, :self._total_classes]
                raw_outputs = logits_main + logits_few
                outputs = raw_outputs.clone()
                if calibration_info is not None:
                    outputs[:, : self._known_classes] *= calibration_info["alpha"]
                if self.concm_uncertainty_diagnostics:
                    unc = self._concm_uncertainty_from_logits(logits_main, logits_few)
                    if unc is not None:
                        ale, epi = unc
                        self._update_uncertainty_ema(ale, epi, targets)
                        y_unc_ale.append(ale.detach().cpu().float().numpy())
                        y_unc_epi.append(epi.detach().cpu().float().numpy())
                        y_uncalibrated_pred.append(raw_outputs.argmax(dim=1).detach().cpu().numpy())
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            y_logits.append(raw_outputs.detach().cpu().float().numpy())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        y_logits = np.concatenate(y_logits)
        if self.concm_uncertainty_diagnostics and len(y_unc_ale) > 0:
            y_unc_ale = np.concatenate(y_unc_ale)
            y_unc_epi = np.concatenate(y_unc_epi)
            y_uncalibrated_pred = np.concatenate(y_uncalibrated_pred)
            self._last_uncertainty_diag = self._build_uncertainty_eval_diag(
                y_true,
                y_pred.T[0],
                y_uncalibrated_pred,
                y_unc_ale,
                y_unc_epi,
            )
        if calibration_info is not None:
            uncalibrated = self._calibrated_eval_metrics(y_logits, y_true, 1.0)
            uncalibrated.pop("predictions")
            calibrated = self._calibrated_eval_metrics(y_logits, y_true, calibration_info["alpha"])
            calibrated.pop("predictions")
            self._last_eval_calibration = {
                "eval_calibration_rule": calibration_info["rule"],
                "eval_calibration_alpha": np.around(calibration_info["alpha"], decimals=6),
                "eval_calibration_old_norm": np.around(calibration_info["old_norm"], decimals=6),
                "eval_calibration_new_norm": np.around(calibration_info["new_norm"], decimals=6),
                "uncalibrated_total": uncalibrated["task1_accT"],
                "uncalibrated_avg_acc": uncalibrated["avg_acc"],
                "uncalibrated_old_acc": uncalibrated["old_acc"],
                "uncalibrated_new_acc": uncalibrated["new_acc"],
                "uncalibrated_new_eval_pred_old_rate": uncalibrated["new_eval_pred_old_rate"],
                "uncalibrated_old_eval_pred_new_rate": uncalibrated["old_eval_pred_new_rate"],
                "calibrated_total": calibrated["task1_accT"],
                "calibrated_avg_acc": calibrated["avg_acc"],
                "calibrated_old_acc": calibrated["old_acc"],
                "calibrated_new_acc": calibrated["new_acc"],
                "calibrated_new_eval_pred_old_rate": calibrated["new_eval_pred_old_rate"],
                "calibrated_old_eval_pred_new_rate": calibrated["old_eval_pred_new_rate"],
            }
            logging.info(
                (
                    "ConCMStage1EvalCalibration rule={} alpha={:.6f} old_norm={:.6f} new_norm={:.6f} "
                    "uncalibrated_total={:.2f} uncalibrated_old_acc={:.2f} uncalibrated_new_acc={:.2f} "
                    "uncalibrated_new_eval_pred_old_rate={:.2f} uncalibrated_old_eval_pred_new_rate={:.2f} "
                    "calibrated_total={:.2f} calibrated_old_acc={:.2f} calibrated_new_acc={:.2f} "
                    "calibrated_new_eval_pred_old_rate={:.2f} calibrated_old_eval_pred_new_rate={:.2f}"
                ).format(
                    calibration_info["rule"],
                    calibration_info["alpha"],
                    calibration_info["old_norm"],
                    calibration_info["new_norm"],
                    uncalibrated["task1_accT"],
                    uncalibrated["old_acc"],
                    uncalibrated["new_acc"],
                    uncalibrated["new_eval_pred_old_rate"],
                    uncalibrated["old_eval_pred_new_rate"],
                    calibrated["task1_accT"],
                    calibrated["old_acc"],
                    calibrated["new_acc"],
                    calibrated["new_eval_pred_old_rate"],
                    calibrated["old_eval_pred_new_rate"],
                )
            )
        self._maybe_save_old_logit_scale_sweep(y_logits, y_true)

        return y_pred, y_true  # [N, topk]

    def _eval_calibration_info(self):
        if (
            not self.concm_stage1_eval_calibration
            or self._cur_task < 1
            or self._known_classes <= 0
        ):
            return None
        if self.calibration_rule != "head_norm_effective_sum":
            raise ValueError("Unsupported calibration_rule: {}".format(self.calibration_rule))

        backbone = self._backbone_module()
        with torch.no_grad():
            head_weight = backbone.head.weight[: self._total_classes].detach()
            head_few_weight = backbone.head_few.weight[: self._total_classes].detach()
            effective_weight = head_weight + head_few_weight
            old_norm = effective_weight[: self._known_classes].norm(dim=1).mean()
            new_norm = effective_weight[self._known_classes : self._total_classes].norm(dim=1).mean()
            alpha = new_norm / old_norm.clamp_min(1e-12)

        return {
            "rule": self.calibration_rule,
            "old_norm": float(tensor2numpy(old_norm)),
            "new_norm": float(tensor2numpy(new_norm)),
            "alpha": float(tensor2numpy(alpha)),
        }

    def _calibrated_eval_metrics(self, logits, labels, alpha):
        calibrated = logits.copy()
        calibrated[:, : self._known_classes] *= alpha
        preds = calibrated.argmax(axis=1)
        old_mask = labels < self._known_classes
        new_mask = labels >= self._known_classes

        total_acc = np.around((preds == labels).sum() * 100 / len(labels), decimals=2)
        old_acc = 0 if not old_mask.any() else np.around((preds[old_mask] == labels[old_mask]).sum() * 100 / old_mask.sum(), decimals=2)
        new_acc = 0 if not new_mask.any() else np.around((preds[new_mask] == labels[new_mask]).sum() * 100 / new_mask.sum(), decimals=2)
        new_eval_pred_old_rate = 0 if not new_mask.any() else np.around((preds[new_mask] < self._known_classes).sum() * 100 / new_mask.sum(), decimals=2)
        old_eval_pred_new_rate = 0 if not old_mask.any() else np.around((preds[old_mask] >= self._known_classes).sum() * 100 / old_mask.sum(), decimals=2)

        if self._previous_eval_top1 is not None:
            avg_acc = np.around((float(self._previous_eval_top1) + float(total_acc)) / 2.0, decimals=3)
        else:
            avg_acc = total_acc

        return {
            "alpha": float(alpha),
            "avg_acc": float(avg_acc),
            "task1_accT": float(total_acc),
            "old_acc": float(old_acc),
            "new_acc": float(new_acc),
            "new_eval_pred_old_rate": float(new_eval_pred_old_rate),
            "old_eval_pred_new_rate": float(old_eval_pred_new_rate),
            "predictions": preds,
        }

    def _maybe_save_old_logit_scale_sweep(self, logits, labels):
        if (
            not bool(self.args.get("eval_old_logit_scale_sweep", False))
            or self._cur_task != 1
            or self._known_classes <= 0
        ):
            return

        alphas = self.args.get("eval_old_logit_scale_alphas", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        alphas = [float(alpha) for alpha in alphas]
        artifact_root = self.args.get("eval_artifact_dir", "eval_artifacts")
        run_name = self.args.get("text", "") or self.args.get("time_str", "run")
        artifact_dir = os.path.join(artifact_root, "{}_{}".format(run_name, self.args.get("time_str", "notime")))
        os.makedirs(artifact_dir, exist_ok=True)

        metrics = []
        predictions_by_alpha = []
        for alpha in alphas:
            metric = self._calibrated_eval_metrics(logits, labels, alpha)
            predictions_by_alpha.append(metric.pop("predictions"))
            metrics.append(metric)
            logging.info(
                "OldLogitScaleSweep alpha={:.2f} AvgAcc={:.3f} task1_AccT={:.2f} old_acc={:.2f} new_acc={:.2f} new_eval_pred_old_rate={:.2f} old_eval_pred_new_rate={:.2f}".format(
                    metric["alpha"],
                    metric["avg_acc"],
                    metric["task1_accT"],
                    metric["old_acc"],
                    metric["new_acc"],
                    metric["new_eval_pred_old_rate"],
                    metric["old_eval_pred_new_rate"],
                )
            )

        old_class_ids = np.arange(0, self._known_classes, dtype=np.int64)
        new_class_ids = np.arange(self._known_classes, self._total_classes, dtype=np.int64)
        predictions_by_alpha = np.stack(predictions_by_alpha, axis=0).astype(np.int64)

        npz_path = os.path.join(artifact_dir, "task{}_old_logit_scale_sweep_logits.npz".format(self._cur_task))
        json_path = os.path.join(artifact_dir, "task{}_old_logit_scale_sweep_metrics.json".format(self._cur_task))
        np.savez_compressed(
            npz_path,
            logits=logits.astype(np.float32),
            labels=labels.astype(np.int64),
            predicted_labels=predictions_by_alpha[0],
            predictions_by_alpha=predictions_by_alpha,
            alphas=np.array(alphas, dtype=np.float32),
            old_class_ids=old_class_ids,
            new_class_ids=new_class_ids,
        )
        with open(json_path, "w") as f:
            json.dump(
                {
                    "task": int(self._cur_task),
                    "known_classes": int(self._known_classes),
                    "total_classes": int(self._total_classes),
                    "previous_eval_top1": None if self._previous_eval_top1 is None else float(self._previous_eval_top1),
                    "metrics": metrics,
                    "logits_npz": npz_path,
                },
                f,
                indent=2,
            )
        logging.info("OldLogitScaleSweep artifacts saved: metrics_json={} logits_npz={}".format(json_path, npz_path))

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct1,correct2, correct3, total = 0, 0, 0, 0
        label_list = []
        cls_num_list = torch.Tensor(self.args["lt_list"][:self._total_classes]).to(self._device)
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                weight = cls_num_list[targets]
                label_list.append(weight.cpu())
                res = model(inputs, task_id=self._cur_task, weight=weight) 
                outputs1 = res["logits"][:, :self._total_classes]
                outputs2= res["logits_few"][:, :self._total_classes] 
                outputs3 = outputs1 + outputs2
                
            predicts1 = torch.max(outputs1, dim=1)[1]
            correct1 += (predicts1.cpu() == targets).sum()

            predicts2 = torch.max(outputs2, dim=1)[1]
            correct2 += (predicts2.cpu() == targets).sum()

            predicts3 = torch.max(outputs3, dim=1)[1]
            correct3 += (predicts3.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct1) * 100 / total, decimals=2), np.around(tensor2numpy(correct2) * 100 / total, decimals=2), np.around(tensor2numpy(correct3) * 100 / total, decimals=2)

    def _evaluate(self, y_pred, y_true):
        ret = super()._evaluate(y_pred, y_true)
        top1 = y_pred.T[0]
        if self._known_classes > 0:
            old_mask = y_true < self._known_classes
            new_mask = y_true >= self._known_classes
            if old_mask.any():
                old_eval_pred_new_rate = np.around((top1[old_mask] >= self._known_classes).sum() * 100 / old_mask.sum(), decimals=2)
                old_to_old_rate = np.around((top1[old_mask] < self._known_classes).sum() * 100 / old_mask.sum(), decimals=2)
            else:
                old_eval_pred_new_rate = 0
                old_to_old_rate = 0
            if new_mask.any():
                new_eval_pred_old_rate = np.around((top1[new_mask] < self._known_classes).sum() * 100 / new_mask.sum(), decimals=2)
                new_to_new_rate = np.around((top1[new_mask] >= self._known_classes).sum() * 100 / new_mask.sum(), decimals=2)
            else:
                new_eval_pred_old_rate = 0
                new_to_new_rate = 0

            ret["grouped"]["new_eval_pred_old_rate"] = new_eval_pred_old_rate
            ret["grouped"]["old_eval_pred_new_rate"] = old_eval_pred_new_rate
            ret["grouped"]["old_to_old_pred_rate"] = old_to_old_rate
            ret["grouped"]["new_to_new_pred_rate"] = new_to_new_rate
            ret["grouped"]["old_new_block_confusion"] = "old->old {:.2f}, old->new {:.2f}, new->old {:.2f}, new->new {:.2f}".format(
                old_to_old_rate,
                old_eval_pred_new_rate,
                new_eval_pred_old_rate,
                new_to_new_rate,
            )

            backbone = self._backbone_module()
            with torch.no_grad():
                head_weight = backbone.head.weight[: self._total_classes].detach()
                head_few_weight = backbone.head_few.weight[: self._total_classes].detach()
                old_head_norm = head_weight[: self._known_classes].norm(dim=1)
                new_head_norm = head_weight[self._known_classes : self._total_classes].norm(dim=1)
                old_head_few_norm = head_few_weight[: self._known_classes].norm(dim=1)
                new_head_few_norm = head_few_weight[self._known_classes : self._total_classes].norm(dim=1)
                ret["grouped"]["head_weight_norm_old_mean"] = np.around(tensor2numpy(old_head_norm.mean()), decimals=6)
                ret["grouped"]["head_weight_norm_new_mean"] = np.around(tensor2numpy(new_head_norm.mean()), decimals=6)
                ret["grouped"]["head_few_weight_norm_old_mean"] = np.around(tensor2numpy(old_head_few_norm.mean()), decimals=6)
                ret["grouped"]["head_few_weight_norm_new_mean"] = np.around(tensor2numpy(new_head_few_norm.mean()), decimals=6)
                effective_head_weight = head_weight + head_few_weight
                old_effective_norm = effective_head_weight[: self._known_classes].norm(dim=1)
                new_effective_norm = effective_head_weight[self._known_classes : self._total_classes].norm(dim=1)
                ret["grouped"]["effective_head_weight_norm_old_mean"] = np.around(tensor2numpy(old_effective_norm.mean()), decimals=6)
                ret["grouped"]["effective_head_weight_norm_new_mean"] = np.around(tensor2numpy(new_effective_norm.mean()), decimals=6)

            if self._last_eval_calibration is not None:
                ret["grouped"].update(self._last_eval_calibration)

            if self._last_proto_calibration_diag:
                ret["grouped"].update(self._last_proto_calibration_diag)

            if self._last_match_anchor_diag:
                ret["grouped"].update(self._last_match_anchor_diag)

            if self._last_usfm_anchor_diag:
                ret["grouped"].update(self._last_usfm_anchor_diag)

            if self._last_uncertainty_diag:
                ret["grouped"].update(self._last_uncertainty_diag)
                ema_diag = self._uncertainty_ema_diag()
                ret["grouped"].update(ema_diag)
                logging.info(
                    (
                        "ConCMUncertaintyEMA task={} old_Ale={} old_Epi={} new_Ale={} new_Epi={} "
                        "many_Ale={} many_Epi={} medium_Ale={} medium_Epi={} few_Ale={} few_Epi={}"
                    ).format(
                        self._cur_task,
                        ema_diag.get("unc_ema_old_ale"),
                        ema_diag.get("unc_ema_old_epi"),
                        ema_diag.get("unc_ema_new_ale"),
                        ema_diag.get("unc_ema_new_epi"),
                        ema_diag.get("unc_ema_many_ale"),
                        ema_diag.get("unc_ema_many_epi"),
                        ema_diag.get("unc_ema_medium_ale"),
                        ema_diag.get("unc_ema_medium_epi"),
                        ema_diag.get("unc_ema_few_ale"),
                        ema_diag.get("unc_ema_few_epi"),
                    )
                )

        self._previous_eval_top1 = float(ret["top1"])
        return ret
