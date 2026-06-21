import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import numpy as np
from parse_config import ConfigParser
import pprint


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=256,
        shuffle=False,
        training=False,
        num_workers=12,
        imb_factor=config['data_loader']['args'].get('imb_factor', 0.01)
    )

    if 'returns_feat' in config['arch']['args']:
        model = config.init_obj('arch', module_arch, allow_override=True, returns_feat=False)
    else:
        model = config.init_obj('arch', module_arch)

    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, weights_only=False)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_metrics = torch.zeros(len(metric_fns))

    num_classes = config._config["arch"]["args"]["num_classes"]
    confusion_matrix = torch.zeros(num_classes, num_classes).cuda()

    get_class_acc = True
    if get_class_acc:
        train_data_loader = getattr(module_data, config['data_loader']['type'])(
            config['data_loader']['args']['data_dir'],
            batch_size=256,
            training=True,
            imb_factor=config['data_loader']['args'].get('imb_factor', 0.01)
        )
        train_cls_num_list = np.array(train_data_loader.cls_num_list)
        many_shot = train_cls_num_list > 100
        medium_shot = (train_cls_num_list <= 100) & (train_cls_num_list >= 20)
        few_shot = train_cls_num_list < 20

    is_afs_model = isinstance(model.module if config['n_gpu'] > 1 else model, module_arch.AFS_ResNet32Model)
    if is_afs_model:
        logger.info("AFS-Model detected. Dynamic gates will be used for testing.")
        real_model = model.module if config['n_gpu'] > 1 else model

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)

            forward_kwargs = {}
            if is_afs_model:
                neutral_gates = torch.zeros((real_model.num_experts, data.size(0), 1), device=device)
                base_result = model(data, gates_per_expert=neutral_gates)
                base_output = base_result['output'] if isinstance(base_result, dict) else base_result

                y_hat = base_output.argmax(dim=1)
                all_gating_tables = torch.stack([c() for c in real_model.controllers],
                                                dim=0)

                y_hat_expanded = y_hat.unsqueeze(1).unsqueeze(0).expand(real_model.num_experts, -1, 1)
                gates_per_expert = torch.gather(all_gating_tables, 1, y_hat_expanded)
                forward_kwargs["gates_per_expert"] = gates_per_expert

            output = model(data, **forward_kwargs)
            output = output['output'] if isinstance(output, dict) else output

            for t, p in zip(target.view(-1), output.argmax(dim=1).view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    n_samples = len(data_loader.dataset)

    acc_per_class = confusion_matrix.diag() / (confusion_matrix.sum(1) + 1e-8)
    acc = acc_per_class.cpu().numpy()

    many_shot_acc = np.nan_to_num(acc[many_shot]).mean()
    medium_shot_acc = np.nan_to_num(acc[medium_shot]).mean()
    few_shot_acc = np.nan_to_num(acc[few_shot]).mean()

    log = {}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })

    if get_class_acc:
        log.update({
            "many_class_num": many_shot.sum(),
            "medium_class_num": medium_shot.sum(),
            "few_class_num": few_shot.sum(),
            "many_shot_acc": many_shot_acc,
            "medium_shot_acc": medium_shot_acc,
            "few_shot_acc": few_shot_acc,
        })
    logger.info(pprint.pprint(log))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume',default = None,
                      type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-l', '--log-config', default='logger/logger_config.json', type=str,
                      help='logging config file path (default: logger/logger_config.json)')

    # dummy arguments used during training time
    args.add_argument("--validate")
    args.add_argument("--use-wandb")

    config, args = ConfigParser.from_args(args)
    main(config)