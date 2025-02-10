import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="config yaml path")
    parser.add_argument(
        "--load", type=str, default=None, help="whether to resume training"
    )
    parser.add_argument(
        "--resume", action="store_true", help="whether to resume training"
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="train",
        help="model runing mode (train/valid/test)",
    )
    parser.add_argument(
        "--valid", action="store_true", help="enable evaluation mode for validation"
    )
    parser.add_argument(
        "--test", action="store_true", help="enable evaluation mode for testset"
    )
    parser.add_argument(
        "--profiler", action="store_true", help="analise execution time"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="enable debug mode for test"
    )
    args = parser.parse_args()
    if args.valid:
        args.mode = "valid"
    elif args.test:
        args.mode = "test"

    return args


def setup_config(cfg, args):
    """
    merge config from yaml file (if any)
    and modify config according to args
    """

    if args.config:
        cfg.merge_from_file(args.config)
    if args.debug:
        opts = ["DEBUG", True, "TRAIN.EPOCHS", 2]
        cfg.merge_from_list(opts)
        args.profiler = True

    exp_name = os.path.split(args.config)[1].replace(".yaml", "")
    experiment_dir = os.path.join(cfg.DIRS.OUTPUTS, exp_name)
    cfg.merge_from_list(
        [
            "DIRS.EXPERIMENT",
            experiment_dir,
            "EXPERIMENT",
            exp_name,
        ]
    )
    cfg.freeze()

    return cfg
