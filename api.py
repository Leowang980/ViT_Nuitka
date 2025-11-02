from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

from deploy import build_parser as build_deploy_parser
from deploy import perform_inference
from encrypt import build_parser as build_encrypt_parser
from encrypt import perform_encryption
from train import build_parser as build_train_parser
from train import run_training

__all__ = ["train_vit", "infer_vit", "encrypt"]


def _namespace_from_parser(
    parser,
    positional: Sequence[str],
    overrides: Dict[str, Any],
):
    # Convert overrides to command-line arguments
    # Need to map Python attribute names (with underscores) to CLI option names (with hyphens)
    cmd_args = list(positional)
    for key, value in overrides.items():
        if isinstance(value, Path):
            value = str(value)
        if value is None:
            continue
        
        # Convert underscore to hyphen for command-line arguments
        # (argparse converts hyphens to underscores for attribute names)
        cli_key = key.replace("_", "-")
        
        if isinstance(value, bool):
            if value:
                cmd_args.append(f"--{cli_key}")
        else:
            cmd_args.extend([f"--{cli_key}", str(value)])
    
    args = parser.parse_args(cmd_args)
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


def encrypt(
    inputs: Union[str, Path, Sequence[Union[str, Path]]],
    **options,
) -> Dict[str, Any]:
    """Encrypt one or more files into a sealed bundle."""
    options = dict(options)
    if isinstance(inputs, (str, Path)):
        positional = [str(inputs)]
    else:
        positional = [str(item) for item in inputs]

    parser = build_encrypt_parser()
    args = _namespace_from_parser(parser, positional, options)
    return perform_encryption(args)
