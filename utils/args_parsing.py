import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="config yaml path")
    parser.add_argument(
        "--resume", action="store_true", help="whether to resume training"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="enable debug mode for test"
    )
    args = parser.parse_args()
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
