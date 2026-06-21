import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from base import BaseTrainer
from utils import inf_loop, MetricTracker, GroupAccuracyTracker
from utils.uncertainty import get_uncertainty_metrics, ClasswiseEMA, softmax_with_temperature
from model.loss import ExpertDecoupleLoss, ExpertDiversityLoss
from model.model import AFS_ResNet32Model

from model.model import DERM_ResNet50Model

class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, combiner,
                 finetuning_combiner=None, valid_data_loader=None, val_criterion=None,
                 lr_scheduler=None, len_epoch=None, save_imgs=False):
        super().__init__(model, criterion, metric_ftns, optimizer, config, val_criterion=val_criterion)
        self.config = config
        self.combiner = combiner
        self.finetuning_combiner = finetuning_combiner
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader) if len_epoch is None else len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.config['trainer'].get('validate', False) and (self.valid_data_loader is not None)
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self._setup_amuse_modules()

        train_metrics = ['loss', 'loss_main', 'loss_decouple', 'loss_diversity'] + [m.__name__ for m in
                                                                                    self.metric_ftns]
        num_experts = config['arch']['args'].get('num_experts', 0)
        return_expert_losses = self.config['loss'].get('return_expert_losses', False)
        if return_expert_losses:
            train_metrics.extend([f'loss_e_{i}' for i in range(num_experts)])
        self.train_metrics = MetricTracker(*train_metrics)

        # # self.is_afs_model = isinstance(self.real_model, AFS_ResNet32Model)
        # # if self.is_afs_model:
        # #     self.logger.info("AFS-Model detected. State update logic will be enabled.")
        #
        # self.is_afs_model = isinstance(self.real_model, DERM_ResNet32Model)
        # if self.is_afs_model:
        #     self.logger.info("DERM-Model detected. State update logic will be enabled.")

        # General check for any model compatible with GUIDE's adaptive policy.
        # The presence of 'controllers' is the defining feature.
        self.is_guide_model = hasattr(self.real_model, 'controllers') and self.real_model.controllers is not None
        if self.is_guide_model:
            self.logger.info("Model detected. Uncertainty-guided policy and TTSA will be enabled.")

        if self.do_validation:
            train_cls_num_list = np.array(data_loader.cls_num_list)
            many_shot = train_cls_num_list > 100
            medium_shot = (train_cls_num_list <= 100) & (train_cls_num_list >= 20)
            few_shot = train_cls_num_list < 20
            val_metrics = ['loss'] + [m.__name__ for m in self.metric_ftns]
            self.valid_metrics = MetricTracker(*val_metrics)
            self.valid_group_acc = GroupAccuracyTracker(many_shot, medium_shot, few_shot, len(train_cls_num_list))

    def _setup_amuse_modules(self):
        cfg = self.config['trainer'].get('amuse_cfg', {})
        self.amuse_cfg = cfg
        # self.logger.info(f"A-MUSE Config: {self.amuse_cfg}")
        self.decouple_criterion = ExpertDecoupleLoss().to(self.device)
        self.lambda_decouple = cfg.get('lambda_decouple', 0.0)
        self.diversity_criterion = ExpertDiversityLoss(T=cfg.get('diversity_temp', 2.0)).to(self.device)
        self.lambda_diversity = cfg.get('lambda_diversity', 0.0)
        if self.do_validation:
            self.unc_ema = ClasswiseEMA(self.config['arch']['args']['num_classes'],
                                        momentum=cfg.get('ema_momentum', 0.9), device=self.device)
            self.gate_cfg = cfg.get('gate_decision', {})
            self.update_gate_every = self.gate_cfg.get('update_every_epochs', 1)
            if self.gate_cfg.get('update_mechanism') == 'meta_optim':
                controller_params = [p for name, p in self.model.named_parameters() if 'controllers' in name]
                if len(controller_params) > 0:
                    self.optimizer_controller = torch.optim.Adam(controller_params, lr=self.gate_cfg.get('lr', 1e-4))
                    self.val_iter = inf_loop(self.valid_data_loader)
                    self.update_gate_every_steps = self.gate_cfg.get('update_every_steps', 200)

    def _train_epoch(self, epoch):
        self.model.train()
        self.real_model._hook_before_iter()
        self.train_metrics.reset()
        if hasattr(self.criterion, "_hook_before_epoch"):
            self.criterion._hook_before_epoch(epoch)
        current_combiner = self._get_combiner(epoch)
        current_combiner.update(epoch)

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            forward_kwargs = self._get_forward_kwargs(target)
            result, loss_dict, acc = current_combiner.forward(self.model, self.criterion, data, target,
                                                              **forward_kwargs)

            loss_main = loss_dict['loss']
            total_loss = loss_main

            feats = result.get('feat')
            logits_orig = result.get('logits')

            loss_decouple = torch.tensor(0.0, device=self.device)
            if feats is not None and self.lambda_decouple > 0:
                loss_decouple = self.decouple_criterion(feats)
                total_loss += self.lambda_decouple * loss_decouple

            loss_diversity = torch.tensor(0.0, device=self.device)
            if logits_orig is not None and self.lambda_diversity > 0:
                logits_for_div = logits_orig.permute(1, 0, 2)
                loss_diversity = self.diversity_criterion(logits_for_div)
                total_loss += self.lambda_diversity * loss_diversity

            total_loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss_main', loss_main.item())
            self.train_metrics.update('loss_decouple', loss_decouple.item())
            self.train_metrics.update('loss_diversity', loss_diversity.item())
            self.train_metrics.update('loss', total_loss.item())
            self.train_metrics.update('accuracy', (acc, len(target)))

            if (batch_idx + 1) % self.log_step == 0:
                self.logger.debug(f'Train Epoch: {epoch} [{batch_idx}/{self.len_epoch}] Loss: {total_loss.item():.6f}')

            if self.gate_cfg.get('update_mechanism') == 'meta_optim':
                if (batch_idx + 1) % self.update_gate_every_steps == 0:
                    self._meta_update_controllers()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.do_validation and self.is_guide_model  and epoch % self.update_gate_every == 0:
            self._epoch_end_meta_update()

        log = self.train_metrics.result()
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
        return log

    def _get_forward_kwargs(self, target=None):
        if self.is_guide_model  and target is not None:
            all_gating_tables = torch.stack([c() for c in self.real_model.controllers], dim=0)
            target_expanded = target.unsqueeze(1).unsqueeze(0).expand(self.real_model.num_experts, -1, 1)
            gates_per_expert = torch.gather(all_gating_tables, 1, target_expanded)
            return {"gates_per_expert": gates_per_expert}
        return {}

    @torch.no_grad()
    def _epoch_end_meta_update(self):
        self.model.eval()
        self.unc_ema = ClasswiseEMA(self.config['arch']['args']['num_classes'],
                                    momentum=self.amuse_cfg.get('ema_momentum', 0.9), device=self.device)

        for data, target in self.valid_data_loader:
            data, target = data.to(self.device), target.to(self.device)

            forward_kwargs = self._get_forward_kwargs(target)
            result = self.model(data, **forward_kwargs)
            logits_orig = result.get('logits')

            if logits_orig is not None:
                probs_per_expert = softmax_with_temperature(logits_orig.permute(1, 0, 2), T=1.0)
                ale, epi = get_uncertainty_metrics(probs_per_expert)
                self.unc_ema.update_batch(ale, epi, target)

        if self.gate_cfg.get('update_mechanism') == 'unc_map':
            stats = self.unc_ema.get_stats()
            ale, epi = stats['ale'], stats['epi']
            alpha = self.gate_cfg.get('alpha', 2.0)
            beta = self.gate_cfg.get('beta', 1.0)
            gamma = self.gate_cfg.get('gamma', 0.0)
            new_logits = alpha * epi - beta * ale + gamma

            for controller in self.real_model.controllers:
                controller.set_gating_logits(new_logits.unsqueeze(-1))
            self.logger.info("Gating logits updated via uncertainty mapping.")

        self.model.train()

    def _meta_update_controllers(self):
        self.model.eval()
        data, target = next(self.val_iter)
        data, target = data.to(self.device), target.to(self.device)

        for name, param in self.model.named_parameters():
            if 'controllers' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        result = self.model(data, **self._get_forward_kwargs(target))
        output = result['output']

        meta_loss = self.criterion(output, target)

        self.optimizer_controller.zero_grad()
        meta_loss.backward()
        self.optimizer_controller.step()

        for param in self.model.parameters():
            param.requires_grad_(True)

        self.model.train()
        self.logger.debug("Meta-updated controllers.")

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        self.valid_group_acc.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                if self.is_guide_model:
                    neutral_gates = torch.zeros((self.real_model.num_experts, data.size(0), 1), device=self.device)
                    base_result = self.model(data, gates_per_expert=neutral_gates)
                    base_output = base_result['output'] if isinstance(base_result, dict) else base_result
                    y_hat = base_output.argmax(dim=1)
                    all_gating_tables = torch.stack([c() for c in self.real_model.controllers], dim=0)
                    y_hat_expanded = y_hat.unsqueeze(1).unsqueeze(0).expand(self.real_model.num_experts, -1, 1)
                    gates_per_expert = torch.gather(all_gating_tables, 1, y_hat_expanded)

                    output = self.model(data, gates_per_expert=gates_per_expert)
                else:
                    output = self.model(data)
                if isinstance(output, dict):
                    output = output["output"]
                loss = self.val_criterion(output, target)
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target, return_length=True))
                self.valid_group_acc.update(target.cpu(), output.cpu())
        log = self.valid_metrics.result()
        per_group_metrics = self.valid_group_acc.accuracy_per_group()
        log['balanced_acc'] = self.valid_group_acc.accuracy(balanced=True)
        log.update(per_group_metrics)
        return log

    def _get_combiner(self, epoch):
        if self.finetuning_combiner is not None and epoch >= self.config['finetuning_combiner']['initial_epoch']:
            return self.finetuning_combiner
        else:
            return self.combiner

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)