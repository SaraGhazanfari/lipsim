import os
import sys
import warnings
import argparse
from os.path import exists, realpath
import submitit
from lipsim.core.trainer import Trainer
from lipsim.core.evaluate import Evaluator

warnings.filterwarnings("ignore")


def override_args(config, depth, num_channels, depth_linear, n_features):
    config.depth = depth
    config.num_channels = num_channels
    config.depth_linear = depth_linear
    config.n_features = n_features
    return config


def set_config(config):
    if config.model_name == 'small':
        config = override_args(config, 20, 45, 7, 2048)  # depth, num_channels, depth_linear, n_features
    elif config.model_name == 'medium':
        config = override_args(config, 30, 60, 10, 2048)
    elif config.model_name == 'large':
        config = override_args(config, 50, 90, 10, 2048)
    elif config.model_name == 'xlarge':
        config = override_args(config, 70, 120, 15, 2048)
    elif config.model_name is None and \
            not all([config.depth, config.num_channels, config.depth_linear, config.n_features]):
        ValueError("Choose --model-name 'small' 'medium' 'large' 'xlarge'")

    # process argments
    eval_mode = ['eval', 'eval_best', 'certified', 'attack']
    os.makedirs('./trained_models', exist_ok=True)
    path = realpath('./trained_models')
    if config.mode == 'train' and config.train_dir is None:
        config.start_new_model = True
        folder = datetime.now().strftime("%Y-%m-%d_%H.%M.%S_%f")[:-2]
        if config.debug:
            folder = 'folder_debug'
            if exists(f'{path}/folder_debug'):
                shutil.rmtree(f'{path}/folder_debug')
        config.train_dir = f'{path}/{folder}'
        os.makedirs(config.train_dir)
        os.makedirs(f'{config.train_dir}/checkpoints')
    elif config.mode == 'train' and config.train_dir is not None:
        config.start_new_model = False
        config.train_dir = f'{path}/{config.train_dir}'
        assert exists(config.train_dir)
    elif config.mode in eval_mode and config.train_dir is not None:
        config.train_dir = f'{path}/{config.train_dir}'
    elif config.mode in eval_mode and config.train_dir is None:
        ValueError("--train_dir must be defined.")
    if config.mode == 'attack' and config.attack is None:
        ValueError('With mode=attack, the following arguments are required: --attack')

    return config


# def main(config):
#     config = set_config(config)
#     if config.mode == 'train':
#         trainer = Trainer(config)
#         trainer()
#     elif config.mode in ['lipsim', 'eval', 'dreamsim']:
#         evaluate = Evaluator(config)
#         return evaluate()

def main(config):
    config = set_config(config)

    ncpus = 20 # 48
    # default: set tasks_per_node equal to number of gpus
    tasks_per_node = config.ngpus
    if config.mode in ['eval', 'eval_best', 'certified', 'attack']:
        tasks_per_node = 1
    cluster = 'slurm' if not config.local else 'local'

    executor = submitit.AutoExecutor(
        folder=config.train_dir, cluster=cluster)
    executor.update_parameters(
        gpus_per_node=config.ngpus,
        nodes=config.nnodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=ncpus // tasks_per_node,
        stderr_to_stdout=True,
        slurm_job_name=f'{config.train_dir[-4:]}_{config.mode}',
        # slurm_constraint=config.constraint,
        slurm_signal_delay_s=0,
        timeout_min=config.timeout,
        #slurm_setup=setup,
    )

    if config.mode == 'train':
        trainer = Trainer(config)
        job = executor.submit(trainer)
        job_id = job.job_id

        # run eval after training
        if not config.no_eval and not config.debug and not config.local:
            config.mode = 'certified'
            executor.update_parameters(
                nodes=1,
                tasks_per_node=1,
                cpus_per_task=20, # todo 40
                slurm_job_name=f'{config.train_dir[-4:]}_{config.mode}',
                slurm_additional_parameters={'dependency': f'afterany:{job_id}'},
                mem='32GB',
                #qos='qos_gpu-t3',
                timeout_min=60
            )
            evaluate = Evaluator(config)
            job = executor.submit(evaluate)
            job_id = job.job_id

    elif config.mode in ['eval', 'eval_best', 'attack', 'certified']:
        evaluate = Evaluator(config)
        job = executor.submit(evaluate)
        job_id = job.job_id

    folder = config.train_dir.split('/')[-1]
    print(f"Submitted batch job {job_id} in folder {folder}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or Evaluate Lipschitz Networks.')

    parser.add_argument("--account", type=str, default='dci@v100',
                        help="Account to use for slurm.")
    parser.add_argument("--ngpus", type=int, default=2,
                        help="Number of GPUs to use.") # todo 4
    parser.add_argument("--nnodes", type=int, default=1,
                        help="Number of nodes.")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Time of the Slurm job in minutes for training.") # todo 1200
    parser.add_argument("--partition", type=str, default="gpu_p13",
                        help="Partition to use for Slurm.")
    parser.add_argument("--qos", type=str, default="qos_gpu-t3",
                        help="Choose Quality of Service for slurm job.")
    parser.add_argument("--constraint", type=str, default=None,
                        help="Add constraint for choice of GPUs: 16 or 32")
    parser.add_argument("--begin", type=str, default='',
                        help="Set time to begin job")
    parser.add_argument("--local", action='store_true',
                        help="Execute with local machine instead of slurm.")
    parser.add_argument("--debug", action="store_true",
                        help="Activate debug mode.")

    # parameters training or eval
    parser.add_argument("--no-eval", action="store_true", default=True, help="No eval after training")
    parser.add_argument("--mode", type=str, default="train",
                        choices=['train', 'certified', 'attack', 'eval', 'dreamsim'])
    parser.add_argument("--train_dir", type=str, help="Name of the training directory.")
    parser.add_argument("--data_dir", type=str, help="Name of the data directory.")
    parser.add_argument("--dataset", type=str, default='imagenet-1k', help="Dataset to use, imagenet-1k, night")

    parser.add_argument("--shift_data", type=bool, default=True, help="Shift dataset with mean.")
    parser.add_argument("--normalize_data", action='store_true', help="Normalize dataset.")

    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs for training.")
    parser.add_argument("--loss", type=str, default="xent", help="Define the loss to use for training.")
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--scheduler", type=str, default="interp")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--wd", type=float, default=0, help="Weight decay to use for training.")
    parser.add_argument("--nesterov", action='store_true', default=False)
    parser.add_argument("--warmup_scheduler", type=float, default=0., help="Percentage of training.")
    parser.add_argument("--decay", type=str, help="Milestones for MultiStepLR")
    parser.add_argument("--gamma", type=float, help="Gamma for MultiStepLR")
    parser.add_argument("--gradient_clip_by_norm", type=float, default=None)
    parser.add_argument("--gradient_clip_by_value", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, help="Make the training deterministic.")
    parser.add_argument("--print_grad_norm", action='store_true', help="Print of the norm of the gradients")
    parser.add_argument("--frequency_log_steps", type=int, default=1000, help="Print log for every step.")
    parser.add_argument("--logging_verbosity", type=str, default='INFO', help="Level of verbosity of the logs")
    parser.add_argument("--save_checkpoint_epochs", type=int, default=1, help="Save checkpoint every epoch.")

    # specific parameters for eval
    parser.add_argument("--attack", type=str, choices=['pgd', 'autoattack'], help="Choose the attack.")
    parser.add_argument("--eps", type=float, default=36)

    # parameters of the architectures
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--depth", type=int, default=30)
    parser.add_argument("--num_channels", type=int, default=30)
    parser.add_argument("--depth_linear", type=int, default=5)
    parser.add_argument("--n_features", type=int, default=2048)
    parser.add_argument("--conv_size", type=int, default=5)
    parser.add_argument("--init", type=str, default='xavier_normal')

    parser.add_argument("--first_layer", type=str, default="padding_channels")
    parser.add_argument("--last_layer", type=str, default="pooling_linear")

    parser.add_argument("--teacher_model_name", type=str, default='dino_vitb16',
                        help='dino_vitb16 open_clip_vitb32 clip_vitb32')

    # parse all arguments
    config = parser.parse_args()
    config.cmd = f"python3 {' '.join(sys.argv)}"

    main(config)
