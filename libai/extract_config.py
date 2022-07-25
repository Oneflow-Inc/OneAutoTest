import math
import argparse

_GLOBAL_ARGS = None


def get_args(extra_args_provider=None):
    global _GLOBAL_ARGS
    if _GLOBAL_ARGS is None:
        _GLOBAL_ARGS = parse_args(extra_args_provider)

    return _GLOBAL_ARGS


def parse_args(extra_args_provider=None, ignore_unknown_args=False):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(
        description="OneFlow ResNet50 DLPerf Arguments", allow_abbrev=False
    )
    parser = _add_training_args(parser)

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    _print_args(args)
    return args


def _print_args(args):
    """Print arguments."""
    print("------------------------ arguments ------------------------", flush=True)
    str_list = []
    for arg in vars(args):
        dots = "." * (48 - len(arg))
        str_list.append("  {} {} {}".format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print("-------------------- end of arguments ---------------------", flush=True)


def _add_training_args(parser):
    group = parser.add_argument_group(title="training")
    group.add_argument("--test-log", type=str, help="log directory")
    group.add_argument("--compare-log", type=str, help="log directory")
    group.add_argument("--oneflow-commit", type=str)

    return parser


if __name__ == "__main__":
    get_args()
