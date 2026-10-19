"""Run the gradient-inversion reconstruction evaluation."""
import os


# limit the visual gpus
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import torch
import torchvision
import torch.nn as nn
import yaml
import numpy as np
import random


import inversefed

torch.backends.cudnn.benchmark = inversefed.consts.BENCHMARK

from collections import defaultdict
import datetime
import time


import json
import hashlib
import csv
import copy
import pickle
import defense
import lpips

import datetime
import logging


def init_logger(output_dir, log_level=logging.INFO):
    """Experiment support utility."""
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    
    fh = logging.FileHandler(os.path.join(output_dir, "main.log"))
    fh.setLevel(log_level)
    fh_formatter = logging.Formatter('%(message)s')  # Only message content
    fh.setFormatter(fh_formatter)
    root_logger.addHandler(fh)

    
    sh = logging.StreamHandler()
    sh.setLevel(log_level)
    sh_formatter = logging.Formatter('%(message)s')  # Only message content
    sh.setFormatter(sh_formatter)
    root_logger.addHandler(sh)

    root_logger.info("-" * 80)

    return root_logger



nclass_dict = {'I32': 1000, 'I64': 1000, 'I128': 1000,
               'CIFAR10': 10, 'CIFAR100': 100, 'CA': 8, 'ImageNet': 1000, 'IMAGENET_IO': 1000,
               'FFHQ': 10, 'FFHQ64': 10, 'FFHQ128': 10, 'OOD_FFHQ': 10, 'OOD_IMAGENET': 1000
               }


parser = inversefed.options()


parser.add_argument('--seed', default=1234, type=float, help='Local learning rate for federated averaging')
parser.add_argument('--batch_size', default=4, type=int, help='Number of mini batch for federated averaging')
parser.add_argument('--local_lr', default=1e-4, type=float, help='Local learning rate for federated averaging')
parser.add_argument('--checkpoint_path', default='', type=str, help='Local learning rate for federated averaging')
parser.add_argument('--gan', default='stylegan2', type=str, help='GAN model option:[stylegan2, biggan]')
parser.add_argument('--config', default='./config_stylegan2', type=str, help='Path of selected config file.')

parser.add_argument('--set', dest='config_overrides', action='append', default=[],
                    metavar='KEY=VALUE', help='Override YAML config; dotted keys are supported.')


args = parser.parse_args()
if args.target_id is None:
    args.target_id = 0
args.save_image = True


defs = inversefed.training_strategy('conservative')
defs.epochs = args.epochs


def load_config(config_path):
    """Load or apply experiment configuration."""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    return config


def apply_config_overrides(config, override_items):
    """Load or apply experiment configuration."""
    
    for item in override_items:
        
        if '=' not in item:
            raise ValueError(f'Invalid --set value: {item!r}; expected KEY=VALUE')
        
        dotted_key, raw_value = item.split('=', 1)
        
        value = yaml.safe_load(raw_value)
        
        keys = dotted_key.split('.')
        
        cursor = config
        
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        
        cursor[keys[-1]] = value
    
    return config


def save_experiment_config(config_path, save_dir, resolved_config=None):
    """Load or apply experiment configuration."""
    
    experiment_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    saved_config_file = os.path.join(save_dir, f"experiment_config_{experiment_id}.yml")

    
    
    if resolved_config is not None:
        config = copy.deepcopy(resolved_config)
    else:
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    with open(saved_config_file, 'w') as file:
        yaml.dump(config, file)

    return saved_config_file


@torch.no_grad()
def evaluate_classification_accuracy(model, dataloader, setup, max_batches=None):
    """Experiment support utility."""
    
    was_training = model.training
    
    model.eval()
    
    correct, total = 0, 0
    
    for batch_index, (inputs, targets) in enumerate(dataloader):
        
        if max_batches is not None and batch_index >= int(max_batches):
            break
        
        inputs = inputs.to(**setup)
        
        targets = targets.to(device=setup['device']).long()
        
        logits = model(inputs)
        
        predictions = logits.argmax(dim=1)
        
        correct += int((predictions == targets).sum().item())
        
        total += int(targets.numel())
    
    model.train(was_training)
    
    return (float(correct) / float(total) if total else float('nan')), total


if __name__ == "__main__":
    
    
    current_time = datetime.datetime.now().strftime("%b.%d_%H.%M.%S")

    start_time = time.time()
    
    config_path = args.config
    config = load_config(config_path=config_path)
    
    config = apply_config_overrides(config, args.config_overrides)
    if config['defense_method'] not in {None, 'noise', 'clipping', 'compression',
            'representation', 'orthogonal', 'gmm', 'gmm_bicone', 'gmm_bicone_v2', 'gmm_budget', 'spectra'}:
        raise ValueError(f"Unsupported defense: {config['defense_method']}")


    
    config['exp_name'] = config['exp_name'] + '_num_images_' + str(config['num_images']) + '_defense_' + str(
        config['defense_method']) + '_' + current_time
    save_dir = os.path.join(config['output_dir'], config['exp_name'])
    os.makedirs(save_dir, exist_ok=True)
    logger = init_logger(save_dir)
    
    pid = os.getpid()
    logging.info(f"Process ID: {pid}")
    logger.info("output_dir: {}".format(save_dir))

    
    setup = inversefed.utils.system_startup(args)

    save_experiment_config(config_path, save_dir, resolved_config=config)
    logger.info(f"Config for this experiment is saved")
    

    loss_fn, trainloader, validloader = inversefed.construct_dataloaders(config['dataset'], defs,
                                                                         data_path=config['data_path'])

    
    set_seed = config['set_seed']
    if isinstance(set_seed, int):
        logger.info("Set seed:{}".format(set_seed))
        torch.manual_seed(set_seed)

    
    model, model_seed = inversefed.construct_model(config['model'], num_classes=nclass_dict[config['dataset']],
                                                   num_channels=3, seed=set_seed)
    model.to(**setup)
    
    
    model_checkpoint_path = config.get('model_checkpoint_path')
    if model_checkpoint_path:
        checkpoint = torch.load(model_checkpoint_path, map_location=setup['device'])
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        state_dict = {key.removeprefix('module.'): value for key, value in state_dict.items()}
        model.load_state_dict(
            state_dict, strict=bool(config.get('model_checkpoint_strict', True)))
        logger.info('Loaded classification model checkpoint: %s', model_checkpoint_path)
    elif bool(config.get('trained_model', False)) or bool(args.trained_model):
        raise ValueError(
            'trained_model=true but model_checkpoint_path is empty; refusing random-model formal evaluation.')

    
    if config['dataset'].startswith('FFHQ') or config['dataset'].endswith('FFHQ'):
        dm = torch.as_tensor(getattr(inversefed.consts, f'cifar10_mean'), **setup)[:, None, None]
        ds = torch.as_tensor(getattr(inversefed.consts, f'cifar10_std'), **setup)[:, None, None]
    else:
        dataset = config['dataset']
        dm = torch.as_tensor(getattr(inversefed.consts, f'{dataset.lower()}_mean'), **setup)[:, None, None]
        ds = torch.as_tensor(getattr(inversefed.consts, f'{dataset.lower()}_std'), **setup)[:, None, None]

    
    # optimizer, scheduler = inversefed.training.set_optimizer(model, defs)
    optimizer = torch.optim.SGD(model.parameters(), lr=defs.lr, momentum=0.9,
                                weight_decay=defs.weight_decay)
    logger.info(f"Optimizer: {optimizer}")

    
    for epoch in range(config['train_epochs']):
        
        save_dir = os.path.join(config['output_dir'], config['exp_name'], f'epoch_{epoch}')
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Epoch {epoch} started,saved at {save_dir}")

        # model = nn.DataParallel(model)
        model.eval()

        
        accuracy_eval_batches = config.get('accuracy_eval_batches', 25)
        acc_before, acc_samples = evaluate_classification_accuracy(
            model, validloader, setup, accuracy_eval_batches)
        logger.info('Validation ACC before protected update: %.6f (%d samples)',
                    acc_before, acc_samples)

        
        if config['optim'] == 'GAN_based':
            config_m = dict(cost_fn=config['cost_fn'],
                            indices=config['indices'],
                            weights=config['weights'],
                            lr=config['lr'] if config['lr'] is not None else 0.1,
                            optim='adam',
                            restarts=config['restarts'],
                            max_iterations=config['max_iterations'],
                            total_variation=config['total_variation'],
                            bn_stat=config['bn_stat'],
                            image_norm=config['image_norm'],
                            z_norm=args.z_norm,
                            group_lazy=config['group_lazy'],
                            init=config['init'],
                            lr_decay=True,
                            dataset=config['dataset'],
                            # params for inter optim
                            ckpt=config['ckpt'],
                            gifd=config['gifd'],
                            steps=config['steps'],
                            lr_io=config['lr_io'],
                            start_layer=config['start_layer'],
                            end_layer=config['end_layer'],
                            project=config.get('project', False),
                            do_project_gen_out=config['do_project_gen_out'],
                            do_project_noises=config['do_project_noises'],
                            do_project_latent=config['do_project_latent'],
                            max_radius_gen_out=config['max_radius_gen_out'],
                            max_radius_noises=config['max_radius_noises'],
                            max_radius_latent=config['max_radius_latent'],
                            # defense
                            defense_method=config['defense_method'],
                            defense_setting=config['defense_setting'],

                            generative_model=config['generative_model'],
                            gen_dataset=config['gen_dataset'],
                            generator_checkpoint_path=config.get('generator_checkpoint_path'),
                            giml='',
                            gias=config['gias'],
                            ggl=config['ggl'],
                            cma_budget=config['cma_budget'],
                            num_sample=config['num_sample'],
                            KLD=config['KLD'],
                            gias_lr=config['gias_lr'],
                            gias_iterations=config['gias_iterations'],
                            )
        
        elif config['optim'] == 'GAN_free':
            config_m = dict(cost_fn=config['cost_fn'],
                            indices=config['indices'],
                            weights=config['weights'],
                            lr=config['lr'] if config['lr'] is not None else 0.1,
                            optim='adam',
                            restarts=config['restarts'],
                            max_iterations=config['max_iterations'],
                            total_variation=config['total_variation'],
                            bn_stat=config['bn_stat'],
                            image_norm=config['image_norm'],
                            z_norm=args.z_norm,
                            group_lazy=config['group_lazy'],
                            init=config['init'],
                            lr_decay=True,
                            dataset=config['dataset'],
                            geiping=config['geiping'],
                            yin=config['yin'],
                            generative_model='',
                            gen_dataset='',
                            giml=False,
                            gias=False,
                            gias_lr=0.0,
                            gias_iterations=0,
                            )

        
        G = None
        if args.checkpoint_path:
            with open(args.checkpoint_path, 'rb') as f:
                G, _ = pickle.load(f)
                G = G.requires_grad_(True).to(setup['device'])

        
        inversefed.utils.save_to_table(os.path.join(config['output_dir'], config['exp_name']), name='configs',
                                       dryrun=args.dryrun, **config)
        target_id = config['target_id']
        iter_dryrun = False
        protected_gradients_for_utility = []
        utility_target_ids = []

        
        for i in range(config['num_exp']):

            
            psnrs = {}
            lpips_sc = {}
            lpips_sc_a = {}
            ssim = {}
            mse_i = {}

            configured_indices = config.get('target_indices')
            if configured_indices is not None:
                if len(configured_indices) != int(config['num_exp']):
                    raise ValueError('The number of target_indices must equal num_exp.')
                target_id = int(configured_indices[i])
            else:
                target_id = config['target_id'] + i * 1000

            tid_list = []
            inferred_label_correct = False

            print(f"Dataset size: {len(validloader.dataset)}")
            print(f"Attempting to access index: {target_id}")

            dataset_size = len(validloader.dataset)
            if target_id < 0 or target_id >= dataset_size:
                raise IndexError(
                    f'target_id={target_id} is outside the dataset range 0--{dataset_size - 1};'
                    'Reduce num_exp or provide valid target indices explicitly.')

            
            if config['num_images'] == 1:
                ground_truth, labels = validloader.dataset[target_id]
                ground_truth, labels = ground_truth.unsqueeze(0).to(**setup), torch.as_tensor((labels,),
                                                                                              device=setup['device'])
                target_id_ = target_id + 1
                logger.info(f"loaded img {target_id_ - 1}")
                tid_list.append(target_id_ - 1)
            else:
                ground_truth, labels = [], []
                target_id_ = target_id
                while len(labels) < config['num_images']:
                    img, label = validloader.dataset[target_id_]
                    target_id_ += 1
                    if (label not in labels):
                        logger.info("loaded img %d" % (target_id_ - 1))
                        labels.append(torch.as_tensor((label,), device=setup['device']))
                        ground_truth.append(img.to(**setup))
                        tid_list.append(target_id_ - 1)

                ground_truth = torch.stack(ground_truth)
                labels = torch.cat(labels)
            
            img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])

            
            if config['bn_stat'] > 0:
                bn_layers = []
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        bn_layers.append(inversefed.BNStatisticsHook(module))

            
            if args.accumulation == 0:
                logger.info("Ground truth's size:{}".format(ground_truth[0].shape))
                target_loss, _, _ = loss_fn(model(ground_truth), labels)
                input_gradient = torch.autograd.grad(target_loss, model.parameters())
                original_input_gradient = tuple(g.detach().clone() for g in input_gradient)

                
                input_gradient_norm = torch.norm(torch.cat([g.view(-1) for g in input_gradient]), p=2)
                
                input_gradient_tmp = input_gradient_norm.cpu().detach().numpy()

                logger.info(f"Input gradient norm: {input_gradient_norm}")
                logger.info(f"Target loss: {target_loss.item()}")

                
                inversefed.utils.save_to_table(os.path.join(save_dir), name=f'{epoch}_epoch_input_gradient_norm',
                                               dryrun=args.dryrun, input_gradient_norm=input_gradient_tmp,
                                               target_loss=target_loss.item(), target_id=target_id, seed=model_seed)

                best_noise_loss = target_loss

                
                bn_prior = []
                if config['bn_stat'] > 0:
                    for idx, mod in enumerate(bn_layers):
                        mean_var = mod.mean_var[0].detach(), mod.mean_var[1].detach()
                        bn_prior.append(mean_var)

                
                if config['defense_method'] is None:
                    logger.info('No defense applied.')
                    d_param = config['defense_setting']
                else:
                    
                    if config['defense_method'] == 'noise':
                        d_param = 0.01 if config['defense_setting']['noise'] is None else config['defense_setting'][
                            'noise']
                        input_gradient = defense.additive_noise(model, input_gradient, save_dir, std=d_param)
                        overhead_end_time = time.time()
                    
                    if config['defense_method'] == 'clipping':
                        d_param = 4 if config['defense_setting']['clipping'] is None else config['defense_setting'][
                            'clipping']
                        input_gradient = defense.gradient_clipping(input_gradient, bound=d_param)
                    
                    if config['defense_method'] == 'compression':
                        d_param = 20 if config['defense_setting']['compression'] is None else config['defense_setting'][
                            'compression']
                        input_gradient = defense.gradient_compression(input_gradient, percentage=d_param)
                    
                    if config['defense_method'] == 'representation':
                        d_param = 10 if config['defense_setting']['representation'] is None else \
                        config['defense_setting']['representation']
                        input_gradient = defense.perturb_representation(input_gradient, model, ground_truth,
                                                                        pruning_rate=d_param)
                    # Original three-stage GMM: estimate manifold, remove
                    # input-reversible directions, then project to safe descent.
                    if str(config['defense_method']).lower() in ('gmm', 'gmm_bicone', 'gmm_bicone_v2'):
                        gmm_cfg = config['defense_setting'].get('gmm', {}) or {}
                        method_name = str(config['defense_method']).lower()
                        use_bicone = (method_name in ('gmm_bicone', 'gmm_bicone_v2')
                                      or bool(gmm_cfg.get('bicone_enabled', False)))
                        use_robust_space = (method_name == 'gmm_bicone_v2'
                                            or bool(gmm_cfg.get('robust_reversible_space', False)))
                        input_gradient, gmm_diagnostics = defense.gmm_gradient(
                            input_gradient, model, ground_truth, labels,
                            tangent_rank=gmm_cfg.get('tangent_rank', 4),
                            fd_epsilon=gmm_cfg.get('fd_epsilon', 0.03),
                            virtual_lr=gmm_cfg.get('virtual_lr', 1e-4),
                            min_descent_cosine=gmm_cfg.get('min_descent_cosine', 0.05),
                            utility_tolerance=gmm_cfg.get('utility_tolerance', 0.02),
                            norm_scales=gmm_cfg.get('norm_scales', [1.0, 0.75, 0.5, 0.25]),
                            seed=gmm_cfg.get('seed', 1234) + epoch,
                            contraction_steps=gmm_cfg.get('contraction_steps', 20),
                            max_residual_tangent_fraction=gmm_cfg.get(
                                'max_residual_tangent_fraction', 0.10 if use_bicone else 1.0),
                            bicone_enabled=use_bicone,
                            bicone_safe_cosine=gmm_cfg.get('bicone_safe_cosine', 0.70),
                            bicone_candidates=gmm_cfg.get('bicone_candidates', 16),
                            bicone_temperature=gmm_cfg.get('bicone_temperature', 0.20),
                            robust_reversible_space=use_robust_space,
                            fd_scales=gmm_cfg.get('fd_scales', [0.5, 1.0, 2.0]
                                                  if use_robust_space else [1.0]),
                            matchability_weight=gmm_cfg.get(
                                'matchability_weight', 1.0 if use_robust_space else 0.0),
                        )
                        best_noise_loss = torch.tensor(
                            gmm_diagnostics['protected_virtual_loss'],
                            device=input_gradient[0].device)
                        d_param = gmm_cfg
                        inversefed.utils.save_to_table(
                            os.path.join(save_dir),
                            name=f'{epoch}_epoch_gmm_diagnostics',
                            dryrun=args.dryrun,
                            **gmm_diagnostics,
                        )
                    # SPECTRA: leakage-sensitive subspace estimation and task-aware spectral attenuation.
                    if str(config['defense_method']).lower() == 'spectra':
                        spectra_cfg = dict(config['defense_setting'].get('spectra', {}) or {})
                        spectra_cfg['seed'] = int(spectra_cfg.get('seed', 1234)) + epoch + int(target_id)
                        input_gradient, spectra_diagnostics = defense.spectra_gradient(
                            input_gradient, model, ground_truth, labels,
                            loss_fn=loss_fn, **spectra_cfg)
                        protected_gradient_snapshot = tuple(
                            gg.detach().clone() for gg in input_gradient)
                        best_noise_loss = torch.tensor(
                            spectra_diagnostics['protected_virtual_loss'],
                            device=input_gradient[0].device)
                        d_param = spectra_cfg
                        inversefed.utils.save_to_table(
                            os.path.join(save_dir), name=f'{epoch}_epoch_spectra_diagnostics',
                            dryrun=args.dryrun, target_id=int(target_id), **spectra_diagnostics)
                    
                    if str(config['defense_method']).lower() == 'gmm_budget':
                        budget_cfg = config['defense_setting'].get('gmm_budget', {}) or {}
                        input_gradient, budget_diagnostics = defense.gmm_budget_gradient(
                            input_gradient, model, ground_truth, labels,
                            tangent_rank=budget_cfg.get('tangent_rank', 2),
                            fd_epsilon=budget_cfg.get('fd_epsilon', 0.03),
                            fd_scales=budget_cfg.get('fd_scales', [0.5, 1.5]),
                            virtual_lr=budget_cfg.get('virtual_lr', 1e-4),
                            utility_budget=budget_cfg.get('utility_budget', 0.02),
                            risk_utility_lambda=budget_cfg.get('risk_utility_lambda', 1.0),
                            boundary_min_alpha=budget_cfg.get('boundary_min_alpha', 0.80),
                            candidate_count=budget_cfg.get('candidate_count', 4),
                            utility_view_count=budget_cfg.get('utility_view_count', 3),
                            utility_noise_std=budget_cfg.get('utility_noise_std', 0.01),
                            norm_jitter=budget_cfg.get('norm_jitter', 0.10),
                            seed=budget_cfg.get('seed', 1234) + epoch,
                        )
                        best_noise_loss = torch.tensor(
                            budget_diagnostics['protected_virtual_loss'],
                            device=input_gradient[0].device)
                        d_param = budget_cfg
                        inversefed.utils.save_to_table(
                            os.path.join(save_dir), name=f'{epoch}_epoch_gmm_budget_diagnostics',
                            dryrun=args.dryrun, **budget_diagnostics)
                    
                    if config['defense_method'] == 'orthogonal' and epoch in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                        logger.info('Orthogonal applied in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].')
                        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9,
                                                    weight_decay=defs.weight_decay)
                        d_param = 1e-4 if config['defense_setting']['orthogonal'] is None else \
                        config['defense_setting']['orthogonal']
                        input_gradient, best_noise_loss = defense.orthogonal_gradient(input_gradient, model,
                                                                                      ground_truth, labels,
                                                                                      trials=config['our_num_tries'],
                                                                                      epsilon=d_param,
                                                                                      best_loss=target_loss)
                    
                    elif config['defense_method'] == 'orthogonal' and epoch not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                        optimizer = torch.optim.SGD(model.parameters(), lr=defs.lr, momentum=0.9,
                                                    weight_decay=defs.weight_decay)
                        logger.info('Orthogonal applied in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], no defense applied later.')

                
                difference_norm = torch.sqrt(sum(
                    (protected.detach() - original.detach()).pow(2).sum()
                    for protected, original in zip(input_gradient, original_input_gradient)))
                original_norm = torch.sqrt(sum(
                    original.detach().pow(2).sum() for original in original_input_gradient))
                relative_change = float((difference_norm / original_norm.clamp_min(1e-12)).item())
                defense_is_noop = bool(
                    config['defense_method'] is not None and
                    relative_change <= float(config.get('defense_min_relative_change', 1e-8)))
                inversefed.utils.save_to_table(
                    os.path.join(save_dir), name=f'{epoch}_epoch_defense_audit',
                    dryrun=args.dryrun, target_id=int(target_id),
                    defense_method=str(config['defense_method']),
                    relative_gradient_change=relative_change,
                    defense_is_noop=int(defense_is_noop), seed=model_seed)
                if defense_is_noop:
                    message = (f"Defense {config['defense_method']} did not change the gradient "
                               f"for target {target_id} (relative change={relative_change:.3e}).")
                    if bool(config.get('fail_on_defense_noop', False)):
                        raise RuntimeError(message)
                    logger.warning(message)
                protected_gradients_for_utility.append(
                    tuple(g.detach().clone() for g in input_gradient))
                
                
                if len(tid_list) != 1:
                    raise RuntimeError(
                        'Per-sample accuracy evaluation requires num_images=1;'
                        f'Current target list: {tid_list}.')
                utility_target_ids.append(int(tid_list[0]))

                
                
                paired_attack_seed = int(config.get('attack_seed', set_seed)) + int(target_id)
                random.seed(paired_attack_seed)
                np.random.seed(paired_attack_seed % (2 ** 32 - 1))
                torch.manual_seed(paired_attack_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(paired_attack_seed)
                logger.info('Paired attack seed reset to %d', paired_attack_seed)

                
                rec_machine = inversefed.GradientReconstructor(model, setup['device'], (dm, ds), config_m,
                                                               num_images=config['num_images'], bn_prior=bn_prior, G=G)

                if G is None:
                    G = rec_machine.G
                logger.info("Real labels:{}".format(labels))
                # SPECTRA end-to-end audit at the exact attack-consumption site.
                if str(config['defense_method']).lower() == 'spectra':
                    attack_norm = torch.sqrt(sum((gg.detach().float() ** 2).sum() for gg in input_gradient))
                    snapshot_norm = torch.sqrt(sum((gg.detach().float() ** 2).sum() for gg in protected_gradient_snapshot))
                    attack_snapshot_diff = torch.sqrt(sum(
                        (aa.detach().float() - bb.detach().float()).pow(2).sum()
                        for aa, bb in zip(input_gradient, protected_gradient_snapshot)))
                    attack_vs_protected = float((attack_snapshot_diff / snapshot_norm.clamp_min(1e-12)).detach().cpu())
                    inversefed.utils.save_to_table(
                        os.path.join(save_dir),
                        name=f'{epoch}_epoch_attack_input_audit',
                        dryrun=args.dryrun,
                        target_id=int(target_id),
                        defense_method='spectra',
                        attack_input_norm=float(attack_norm.detach().cpu()),
                        attack_vs_protected_relative_change=attack_vs_protected,
                        accumulation=int(args.accumulation),
                        seed=int(config.get('seed', 0)),
                    )
                result = rec_machine.reconstruct(input_gradient, labels, img_shape=img_shape, dryrun=iter_dryrun)
                inferred_labels = getattr(rec_machine, 'last_inferred_labels', None)
                inferred_label_correct = bool(
                    inferred_labels is not None and
                    torch.equal(inferred_labels.detach().view(-1).cpu(), labels.detach().view(-1).cpu()))
                if iter_dryrun:
                    continue
            else:
                
                local_gradient_steps = args.accumulation
                local_lr = args.local_lr
                batch_size = args.batch_size
                input_parameters = inversefed.reconstruction_algorithms.loss_steps(model, ground_truth,
                                                                                   labels,
                                                                                   lr=local_lr,
                                                                                   local_steps=local_gradient_steps,
                                                                                   use_updates=True,
                                                                                   batch_size=batch_size)
                input_parameters = [p.detach() for p in input_parameters]

                rec_machine = inversefed.FedAvgReconstructor(model, (dm, ds), local_gradient_steps,
                                                             local_lr, config_m,
                                                             num_images=config['num_images'], use_updates=True,
                                                             batch_size=batch_size)
                if G is None:
                    if rec_machine.generative_model_name in ['stylegan2']:
                        G = rec_machine.G_synthesis
                    else:
                        G = rec_machine.G
                result = rec_machine.reconstruct(input_parameters, labels, img_shape=img_shape, dryrun=args.dryrun)

            
            lpips_loss = lpips.LPIPS(net='vgg', spatial=False).to(**setup)
            lpips_loss_a = lpips.LPIPS(net='alex', spatial=False).to(**setup)

            
            Best_layer_num = -1
            for idx, item in enumerate(result):
                # Compute stats and save to a table:
                file_name = item[0]
                output = item[1]
                stats = item[2]

                if file_name == "Best_layer_num":
                    Best_layer_num = int(output)
                    continue

                
                if output is None:  # some layers were skipped
                    test_psnr = -1
                    lpips_score = -1
                    lpips_score_a = -1
                    ssim_score = -1
                    test_mse = -1
                    feat_mse = -1
                elif output.shape[-1] != ground_truth.shape[-1]:
                    test_psnr = -1
                    lpips_score = -1
                    lpips_score_a = -1
                    ssim_score = -1
                    test_mse = -1
                    feat_mse = -1
                    output_den = torch.clamp(output * ds + dm, 0, 1)
                else:
                    
                    output_den = torch.clamp(output * ds + dm, 0, 1)
                    ground_truth_den = torch.clamp(ground_truth * ds + dm, 0, 1)
                    # logger.info("output's dimension:{} ground_truth's dimension:{}".format(output.shape, ground_truth.shape))
                    feat_mse = (model(output) - model(ground_truth)).pow(2).mean().item()
                    test_mse = (output_den - ground_truth_den).pow(2).mean().item()
                    
                    ssim_score, _ = inversefed.metrics.ssim_batch(output_den, ground_truth_den)
                    output_lpips = output_den.mul(2).sub(1)
                    ground_truth_lpips = ground_truth_den.mul(2).sub(1)
                    with torch.no_grad():
                        lpips_score = lpips_loss(output_lpips, ground_truth_lpips).squeeze().mean().item()
                        lpips_score_a = lpips_loss_a(output_lpips, ground_truth_lpips).squeeze().mean().item()
                    logger.info("output_den's dimension:{}".format(output_den.shape))
                    test_psnr_value = inversefed.metrics.psnr(output_den, ground_truth_den, factor=1)
                    test_psnr = float(test_psnr_value.detach().cpu().item()
                                      if torch.is_tensor(test_psnr_value) else test_psnr_value)
                    logger.info(
                        f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | LPIPS(VGG): {lpips_score:2.4f} | LPIPS(ALEX): {lpips_score_a:2.4f} | SSIM: {ssim_score:2.4f} | PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} | ")

                validity_cfg = config.get('attack_validity', {}) or {}
                metrics_finite = bool(np.isfinite([
                    test_mse, test_psnr, ssim_score, lpips_score, lpips_score_a]).all())
                reconstruction_valid = bool(
                    metrics_finite and test_mse >= 0 and test_psnr >= 0)
                baseline_converged = bool(
                    reconstruction_valid and inferred_label_correct and
                    test_mse <= float(validity_cfg.get('max_mse', 0.25)) and
                    test_psnr >= float(validity_cfg.get('min_psnr', 6.0)))
                
                
                eligible_for_ranking = (baseline_converged
                                        if config['defense_method'] is None else None)

                ouput_dir = os.path.join(save_dir, file_name)

                psnrs[file_name + '_psnr'] = test_psnr
                lpips_sc[file_name + '_lpips(vgg)'] = lpips_score
                lpips_sc_a[file_name + '_lpips(alex)'] = lpips_score_a
                ssim[file_name + '_ssim'] = ssim_score
                mse_i[file_name + '_mse_i'] = test_mse

                os.makedirs(os.path.join(ouput_dir), exist_ok=True)

                exp_name = config['exp_name']
                inversefed.utils.save_to_table(os.path.join(ouput_dir), name=f'{exp_name}', dryrun=args.dryrun,
                                               rec_loss=stats["opt"],
                                               psnr=test_psnr,
                                               LPIPS_VGG=lpips_score,
                                               LPIPS_ALEX=lpips_score_a,
                                               ssim=ssim_score,
                                               test_mse=test_mse,
                                               feat_mse=feat_mse,

                                               metric_domain='pixel_01_lpips_minus1_1',

                                               metrics_finite=int(metrics_finite),
                                               inferred_label_correct=int(inferred_label_correct),
                                               baseline_converged=(int(baseline_converged)
                                                                   if config['defense_method'] is None else ''),
                                               eligible_for_ranking=(int(eligible_for_ranking)
                                                                     if eligible_for_ranking is not None else ''),

                                               target_id=target_id,
                                               seed=model_seed
                                               )

                
                if args.save_image and output is not None:
                    for j in range(config['num_images']):
                        torchvision.utils.save_image(output_den[j:j + 1, ...],
                                                     os.path.join(ouput_dir, f'{tid_list[j]}_gen.png'))
                
                target_id = target_id_

            
            if Best_layer_num >= 0:
                inversefed.utils.save_to_table(os.path.join(save_dir), name='Metrics', dryrun=args.dryrun,
                                               target_id=int(target_id - 1), Best_layer_num=Best_layer_num, **psnrs,
                                               **lpips_sc, **lpips_sc_a, **ssim, **mse_i)
            else:
                inversefed.utils.save_to_table(os.path.join(save_dir), name='Metrics', dryrun=args.dryrun,
                                               target_id=int(target_id - 1), **psnrs, **lpips_sc, **lpips_sc_a, **ssim,
                                               **mse_i)

            
            for j in range(config['num_images']):
                torchvision.utils.save_image(ground_truth_den[j:j + 1, ...],
                                             os.path.join(save_dir, f'{tid_list[j]}_gt.png'))
            # one row represents psnrs of a batch

        
        
        learning_rate = float(config.get('utility_update_lr', 0.001))
        if not protected_gradients_for_utility:
            raise RuntimeError('No protected gradients are available for utility evaluation.')
        if len(protected_gradients_for_utility) != len(utility_target_ids):
            raise RuntimeError('The number of protected gradients does not match target_id.')

        baseline_parameters = tuple(param.detach().clone() for param in model.parameters())
        sample_acc_after = []
        for sample_target_id, sample_gradient in zip(
                utility_target_ids, protected_gradients_for_utility):
            with torch.no_grad():
                for param, baseline, protected_grad in zip(
                        model.parameters(), baseline_parameters, sample_gradient):
                    param.copy_(baseline - learning_rate * protected_grad)
            sample_acc, sample_acc_count = evaluate_classification_accuracy(
                model, validloader, setup, accuracy_eval_batches)
            sample_acc_after.append(sample_acc)
            inversefed.utils.save_to_table(
                os.path.join(save_dir), name=f'{epoch}_epoch_accuracy',
                dryrun=args.dryrun, target_id=sample_target_id,
                acc_before=acc_before, acc_after=sample_acc,
                acc_delta=sample_acc - acc_before,
                accuracy_samples=min(acc_samples, sample_acc_count),
                utility_gradient_samples=1,
                accuracy_scope='per_sample_independent_update',
                defense_method=str(config['defense_method']), seed=model_seed)

        
        
        mean_protected_gradient = tuple(
            torch.stack([sample_gradient[index] for sample_gradient in protected_gradients_for_utility]).mean(0)
            for index in range(len(protected_gradients_for_utility[0])))
        with torch.no_grad():
            for param, baseline, best_grad in zip(
                    model.parameters(), baseline_parameters, mean_protected_gradient):
                param.copy_(baseline - best_grad * learning_rate)
            logger.info("Model updated with the mean protected gradient from %d samples.",
                        len(protected_gradients_for_utility))

        
        acc_after, acc_samples_after = evaluate_classification_accuracy(
            model, validloader, setup, accuracy_eval_batches)
        
        inversefed.utils.save_to_table(
            os.path.join(save_dir), name=f'{epoch}_epoch_run_accuracy',
            dryrun=args.dryrun, acc_before=acc_before,
            acc_after=acc_after, acc_delta=acc_after - acc_before,
            accuracy_samples=min(acc_samples, acc_samples_after),
            utility_gradient_samples=len(protected_gradients_for_utility),
            sample_acc_mean=float(np.mean(sample_acc_after)),
            sample_acc_std=float(np.std(sample_acc_after, ddof=1))
            if len(sample_acc_after) > 1 else 0.0,
            accuracy_scope='mean_gradient_single_update',
            defense_method=str(config['defense_method']), seed=model_seed)
        logger.info('Validation ACC after protected update: %.6f; delta: %+.6f',
                    acc_after, acc_after - acc_before)

        
        noisy_input_gradient_norm = torch.norm(torch.stack([g.norm() for g in input_gradient]), 2)
        logger.info('Best noise gradient L2 norm: {}'.format(noisy_input_gradient_norm))
        inversefed.utils.save_to_table(os.path.join(save_dir), name=f'{epoch}_epoch_noise_gradient_norm',
                                       dryrun=args.dryrun,
                                       noisy_input_gradient_norm=str(noisy_input_gradient_norm.item()),
                                       best_noise_loss=best_noise_loss.item(), target_id=target_id, seed=model_seed)

        
        post_attack_train_batches = config.get('post_attack_train_batches')
        model.train()
        for i, (inputs, targets) in enumerate(trainloader):
            if post_attack_train_batches is not None and i >= int(post_attack_train_batches):
                break
            logger.info(f"Epoch {epoch} batch {i} started")
            optimizer.zero_grad()
            inputs = inputs.to(**setup)
            targets = targets.to(**setup)
            targets = targets.long()
            outputs = model(inputs)
            loss, _, _ = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            logger.info(f"loss: {loss.item()} at epoch {epoch} batch {i}")

        if post_attack_train_batches != 0:
            logger.info(f"Epoch {epoch} training loss: {loss.item()}")

        
        if bool(config.get('save_model_checkpoint', False)):
            save_path = os.path.join(save_dir, f"model_epoch_{epoch}.pt")
            temporary_path = save_path + '.tmp'
            torch.save(model.state_dict(), temporary_path)
            os.replace(temporary_path, save_path)
            logger.info(f"Model {epoch} epoch checkpoint saved at {save_path}")

    
    logger.info(datetime.datetime.now().strftime("%A, %d %B %Y %I:%M%p"))
    logger.info('---------------------------------------------------')
    logger.info(f'Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}')
    logger.info("output_dir: {}".format(save_dir))
    logger.info('-------------Job finished.-------------------------')
