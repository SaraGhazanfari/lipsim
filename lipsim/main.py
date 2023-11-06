import os
import shutil
import sys
import warnings
import argparse
from datetime import datetime
from os.path import exists, realpath
import submitit

from lipsim.core.evaluate import Evaluator
from lipsim.core.trainer import Trainer
from lipsim.eval_linear import LinearEvaluation

# from lipsim.core.evaluate import Evaluator

warnings.filterwarnings("ignore")

eval_mode = ['certified', 'attack', 'eval', 'ssa', 'lipsim', 'classifier', 'knn']
all_mode = ['train', 'finetune']
all_mode.extend(eval_mode)


def override_args(config, depth, num_channels, depth_linear, n_features):
    config.depth = depth
    config.num_channels = num_channels
    config.depth_linear = depth_linear
    config.n_features = n_features
    return config


def set_config(config):
    if config.model_name == 'small':
        config = override_args(config, 20, 45, 7, 1024)  # depth, num_channels, depth_linear, n_features
    elif config.model_name == 'medium':
        config = override_args(config, 30, 60, 10, 2048)
    elif config.model_name == 'large':
        config = override_args(config, 50, 90, 10, 2048)
    elif config.model_name == 'xlarge':
        config = override_args(config, 70, 120, 15, 2048)
    elif config.model_name is None and \
            not all([config.depth, config.num_channels, config.depth_linear, config.n_features]):
        ValueError("Choose --model-name 'small' 'medium' 'large' 'xlarge'")

    # cluster constraint
    if config.constraint and config.partition != 'gpu_p5':
        config.constraint = f"v100-{config.constraint}g"

    # process argments

    if config.data_dir is None:
        config.data_dir = os.environ.get('DATADIR', None)
    if config.data_dir is None:
        ValueError("the following arguments are required: --data_dir")
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
    elif config.mode in ['train', 'finetune'] and config.train_dir is not None:
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


def main(config):
    config = set_config(config)

    ncpus = 20
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
        exclusive=True,
        # slurm_account=config.account,
        slurm_job_name=f'{config.train_dir[-4:]}_{config.mode}',
        # slurm_partition=config.partition,
        # slurm_qos=config.qos,
        slurm_constraint=config.constraint,
        slurm_signal_delay_s=0,
        timeout_min=config.timeout,
    )
    if config.mode == 'train':

        trainer = Trainer(config)
        job = executor.submit(trainer)
        job_id = job.job_id
        folder = config.train_dir.split('/')[-1]
        print(f"Submitted batch job {job_id} in folder {folder}")
    elif config.mode == 'finetune':
        trainer = Trainer(config)
        trainer.finetune_func()
    elif config.mode == 'classifier':
        linear_eval = LinearEvaluation(config)
        job = executor.submit(linear_eval)
        job_id = job.job_id
        folder = config.train_dir.split('/')[-1]
        print(f"Submitted batch job {job_id} in folder {folder}")
    elif config.mode in eval_mode:
        evaluate = Evaluator(config)
        model = evaluate()
        return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or Evaluate Lipschitz Networks.')

    parser.add_argument("--account", type=str, default='dci@v100',
                        help="Account to use for slurm.")
    parser.add_argument("--ngpus", type=int, default=4,
                        help="Number of GPUs to use.")
    parser.add_argument("--nnodes", type=int, default=1,
                        help="Number of nodes.")
    parser.add_argument("--timeout", type=int, default=1200,
                        help="Time of the Slurm job in minutes for training.")  # 1440
    parser.add_argument("--partition", type=str, default="gpu_p13",
                        help="Partition to use for Slurm.")
    parser.add_argument("--qos", type=str, default="qos_gpu-t3",
                        help="Choose Quality of Service for slurm job.")
    parser.add_argument("--constraint", type=str, default=None,
                        help="Add constraint for choice of GPUs: 16 or 32")
    parser.add_argument("--local", action='store_true',
                        help="Execute with local machine instead of slurm.")
    parser.add_argument("--debug", action="store_true",
                        help="Activate debug mode.")

    # parameters training or eval
    parser.add_argument("--mode", type=str, default="train", choices=all_mode)
    parser.add_argument("--train_dir", type=str, help="Name of the training directory.")
    parser.add_argument("--data_dir", type=str, help="Name of the data directory.")
    parser.add_argument("--dataset", type=str, default='imagenet', help="Dataset to use")

    parser.add_argument("--shift_data", type=bool, default=True, help="Shift dataset with mean.")
    parser.add_argument("--normalize_data", action='store_true', help="Normalize dataset.")

    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training.")
    # parser.add_argument("--loss", type=str, default="xent", help="Define the loss to use for training.")
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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, help="Make the training deterministic.")
    parser.add_argument("--print_grad_norm", action='store_true', help="Print of the norm of the gradients")
    parser.add_argument("--frequency_log_steps", type=int, default=100, help="Print log for every step.")
    parser.add_argument("--logging_verbosity", type=str, default='INFO', help="Level of verbosity of the logs")
    parser.add_argument("--save_checkpoint_epochs", type=int, default=1, help="Save checkpoint every epoch.")

    # specific parameters for eval
    parser.add_argument("--attack", type=str, choices=['PGD-L2', 'PGD-Linf', 'AA-L2', 'AA-Linf', 'CW-L2', 'CW-Linf'],
                        help="Choose the attack.")
    parser.add_argument("--eps", type=float, default=36)

    # parameters of the architectures
    parser.add_argument("--model-name", type=str, default='small')
    parser.add_argument("--depth", type=int, default=30)
    parser.add_argument("--num_channels", type=int, default=30)
    parser.add_argument("--depth_linear", type=int, default=5)
    parser.add_argument("--n_features", type=int, default=2048)
    parser.add_argument("--conv_size", type=int, default=5)
    parser.add_argument("--init", type=str, default='xavier_normal')

    parser.add_argument("--first_layer", type=str, default="padding_channels")
    parser.add_argument("--last_layer", type=str, default="pooling_linear")

    parser.add_argument("--path_embedding", type=str,
                        default='/gpfsscratch/rech/dci/uuc79vj/imagenet_dreamsim')
    parser.add_argument("--dreamsim_path", type=str,
                        default='/gpfswork/rech/dci/uuc79vj/lipsim/dreamsim_ckpts')
    parser.add_argument("--teacher_model_name", type=str, default='ensemble',
                        help='dino_vitb16 open_clip_vitb32 clip_vitb32')

    parser.add_argument("--margin", type=float, default=0)

    # parse all arguments
    config = parser.parse_args()
    config.cmd = f"python3 {' '.join(sys.argv)}"

    main(config)
