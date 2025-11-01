from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

from deploy import build_parser as build_deploy_parser
from deploy import perform_inference
from train import build_parser as build_train_parser
from train import run_training

__all__ = ["train_vit", "infer_vit"]


def _namespace_from_parser(
    parser,
    positional: Sequence[str],
    overrides: Dict[str, Any],
):
    args = parser.parse_args(list(positional))
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise ValueError(f"Unknown option '{key}' for parser.")
        if isinstance(value, Path):
            value = str(value)
        setattr(args, key, value)
    return args


def train_vit(**options) -> float:
    """Run Vision Transformer training with optional overrides.

    Returns the best validation accuracy observed during training.
    """
    options = dict(options)
    parser = build_train_parser()
    args = _namespace_from_parser(parser, [], options)
    return run_training(args)


def infer_vit(
    inputs: Union[str, Path, Sequence[Union[str, Path]]],
    **options,
) -> Dict[str, List[Tuple[str, float]]]:
    """Run inference on one or more images and return top-k predictions."""
    options = dict(options)
    if isinstance(inputs, (str, Path)):
        positional = [str(inputs)]
    else:
        positional = [str(item) for item in inputs]

    parser = build_deploy_parser()
    options.setdefault("quiet", True)
    args = _namespace_from_parser(parser, positional, options)
    return perform_inference(args)
