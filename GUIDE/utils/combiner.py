import numpy as np
import torch, math
from model.metric import accuracy


class Combiner:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.num_epochs = cfg["trainer"]["epochs"]
        self.activation = torch.nn.Softmax(dim=1)
        self.initilize_all_parameters()

    def initilize_all_parameters(self):
        self.mode = self.cfg["combiner"]["mode"]
        self.mixup_alpha = self.cfg["combiner"]["mixup"].get("alpha", 0.0)
        self.cutmix_alpha = self.cfg["combiner"]["mixup"].get("cutmix_alpha", 0.0)
        self.target_mix_strategy = self.cfg["combiner"]["mixup"].get("target_mix_strategy", "mix_input")
        assert self.target_mix_strategy in ["mix_input", "mix_logits"]
        print('_' * 100)
        print('combiner type: ', self.mode)
        print('mixup alpha in combiner: ', self.mixup_alpha)
        print('cutmix alpha in combiner: ', self.cutmix_alpha)
        print('_' * 100)

    def update(self, epoch):
        self.epoch = epoch

    def forward(self, model, criterion, image, label, **kwargs):
        return eval("self.{}".format(self.mode))(
            model, criterion, image, label, **kwargs
        )

    def default(self, model, criterion, image, label, target_model=None, **kwargs):
        result = model(image, **kwargs)
        output = self.get_output(result)

        loss_extra_params = self._get_extra_loss_params(result)
        if target_model is not None:
            with torch.no_grad():
                loss_extra_params['extra_info']['target'] = target_model(image, **kwargs)

        loss = criterion(output, label, **loss_extra_params)
        loss_dict = self._as_loss_dict(loss)
        now_result = self.activation(output)
        now_acc = accuracy(now_result, label)
        return result, loss_dict, now_acc

    def mixup(self, model, criterion, image, label, target_model=None, **kwargs):
        lambda_ = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        idx = torch.randperm(image.size(0))
        image_a, image_b = image, image[idx]
        label_a, label_b = label, label[idx]

        mixed_image = lambda_ * image_a + (1 - lambda_) * image_b
        result = model(mixed_image, **kwargs)

        loss_extra_params = self._get_extra_loss_params(result)
        with torch.no_grad():
            if self.target_mix_strategy == 'mix_logits':
                if target_model is None:
                    target_model = model

                target_dict = self.get_extra_info(target_model(image, **kwargs))
                target_out = target_dict['output']
                target_out_a, target_out_b = target_out, target_out[idx]
                mixed_target_out = lambda_ * target_out_a + (1 - lambda_) * target_out_b
                target_dict['output'] = mixed_target_out

                target_logits = target_dict['logits']
                if target_logits is not None:
                    target_logits_a, target_logits_b = target_logits, target_logits[idx, :, :]
                    mixed_target_logits = lambda_ * target_logits_a + (1 - lambda_) * target_logits_b
                    target_dict['logits'] = mixed_target_logits

                loss_extra_params['extra_info']['target'] = target_dict
            else:
                if target_model is not None:
                    loss_extra_params['extra_info']['target'] = self.get_extra_info(target_model(mixed_image, **kwargs))

        output = self.get_output(result)
        loss_a = criterion(output, label_a, **loss_extra_params)
        loss_b = criterion(output, label_b, **loss_extra_params)

        loss_a_dict, loss_b_dict = self._as_loss_dict(loss_a), self._as_loss_dict(loss_b)
        loss_dict = {}
        for loss_k in loss_a_dict.keys():
            loss_dict[loss_k] = self._mix_values(lambda_, loss_a_dict[loss_k], loss_b_dict[loss_k])

        now_result = self.activation(output)
        now_acc = self._mix_values(lambda_, accuracy(now_result, label_a), accuracy(now_result, label_b))
        return result, loss_dict, now_acc

    def cutmix(self, model, criterion, image, label, target_model=None, **kwargs):
        lambda_ = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        idx = torch.randperm(image.size(0))
        image_a, image_b = image, image[idx]
        label_a, label_b = label, label[idx]

        mixed_image = image_a
        batch_size, _, W, H = mixed_image.size()
        bbx1, bby1, bbx2, bby2 = rand_bbox(mixed_image.size(), lambda_)
        mixed_image[:, :, bbx1:bbx2, bby1:bby2] = image_b[:, :, bbx1:bbx2, bby1:bby2]
        lambda_ = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        result = model(mixed_image, **kwargs)

        loss_extra_params = self._get_extra_loss_params(result)
        if target_model is not None:
            with torch.no_grad():
                loss_extra_params['extra_info']['target'] = target_model(mixed_image, **kwargs)

        output = self.get_output(result)
        loss_a = criterion(output, label_a, **loss_extra_params)
        loss_b = criterion(output, label_b, **loss_extra_params)

        loss_a_dict, loss_b_dict = self._as_loss_dict(loss_a), self._as_loss_dict(loss_b)
        loss_dict = {}
        for loss_k in loss_a_dict.keys():
            loss_dict[loss_k] = self._mix_values(lambda_, loss_a_dict[loss_k], loss_b_dict[loss_k])

        now_result = self.activation(output)
        now_acc = self._mix_values(lambda_, accuracy(now_result, label_a), accuracy(now_result, label_b))
        return result, loss_dict, now_acc

    def mixup_cutmix(self, model, criterion, image, label, mixup_prob=0.5, **kwargs):
        if np.random.binomial(n=1, p=mixup_prob):
            return self.mixup(model, criterion, image, label, **kwargs)
        else:
            return self.cutmix(model, criterion, image, label, **kwargs)

    def get_output(self, result):
        if isinstance(result, dict):
            output = result['output']
        else:
            output = result
        return output

    def get_extra_info(self, result):
        assert self.cfg["add_extra_info"]

        if self.cfg["arch"]["args"]["num_experts"] == 1:
            logits = self.get_output(result).unsqueeze(1)
        else:
            assert isinstance(result, dict)
            logits = result["logits"]

        feat = result.get('feat')

        return dict(
            logits=logits,
            feat=feat,
            output=self.get_output(result),
            epoch=self.epoch,
            num_epochs=self.num_epochs
        )

    def _mix_values(self, lambda_, val1, val2):
        return lambda_ * val1 + (1 - lambda_) * val2

    def _as_loss_dict(self, loss):
        if isinstance(loss, (tuple, list)):
            loss, loss_dict = loss
            assert isinstance(loss_dict, dict)
            loss_dict['loss'] = loss
        else:
            loss_dict = dict(loss=loss)
        return loss_dict

    def _get_extra_loss_params(self, result):
        loss_extra_params = dict(extra_info=self.get_extra_info(result)) if self.cfg["add_extra_info"] else {}
        return_expert_losses = self.cfg['loss'].get('return_expert_losses', False)
        if return_expert_losses:
            loss_extra_params['return_expert_losses'] = return_expert_losses
        return loss_extra_params


def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2