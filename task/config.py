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
        description="OneBrain Task Arguments", allow_abbrev=False
    )
    parser = _add_args(parser)

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    return args


def _add_args(parser):
    group = parser.add_argument_group(title="onebrain")

    group.add_argument(
        "--onebrain-server",
        type=str,
        default="http://180.76.152.239:8081",
        help="onebrain server",
    )
    group.add_argument("--grant-type", type=str, default="cli", help="grant type")
    group.add_argument("--client-id", type=str, help="client id")
    group.add_argument("--client-secret", type=str, help="client secret")
    group.add_argument(
        "--public-key", type=str, help="public key base64.standard_b64encode"
    )
    group.add_argument("--onebrain-username", type=str, help="onebrain username")
    group.add_argument("--onebrain-password", type=str, help="onebrain password")
    group.add_argument("--onebrain-project-id", type=str, help="onebrain project id")

    return parser


if __name__ == "__main__":
    get_args()
