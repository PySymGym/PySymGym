import argparse
import json
import pathlib
import typing as t
import warnings
from dataclasses import dataclass

import onnx
import onnxruntime
import torch
import yaml
from torch.jit import TracerWarning
from common.game import GameState
from ml.dataset import convert_input_to_tensor
from ml.inference import ONNX, TORCH
from torch_geometric import backend as pyg_backend
from torch_geometric.data import HeteroData
from ml.inference import infer
from ml.models.registry import get_model_class

# working version
ONNX_OPSET_VERSION = 17


@dataclass(frozen=True)
class ModelInput:
    """One input of the ONNX model.

    The single source of the model's input contract (#597): the tensor
    builders, the legacy exporter's ``dynamic_axes`` mapping, the dynamo
    exporter's ``dynamic_shapes`` mapping, and the ``input_names`` list are
    all derived from ``MODEL_INPUTS``, so the 11 inputs can no longer drift
    apart.
    """

    #: ONNX wire name (the V# contract; see ``ml.inference.ONNX``)
    name: str
    #: HeteroData key: a node-type str or an edge-type (src, rel, dst) tuple
    torch_key: t.Union[str, t.Tuple[str, str, str]]
    #: Attribute of the HeteroData entry holding the tensor
    attr: str
    #: Dynamic axes: axis index -> symbolic dimension name; absent axes are
    #: static. Axis names are required: torch.onnx.utils._validate_dynamic_axes
    #: warns once per input whose value is an axis list ("No names were found
    #: for specified dynamic axes") and auto-generates the dim params
    #: instead. Names must be unique per distinct size — ONNX treats equal
    #: dim_param strings as the same symbolic dimension — except that an edge
    #: index's axis 1 and the matching edge type's (or per-edge attributes')
    #: axis 0 always hold the same number of edges, so they share a name.
    dynamic_axes: t.Mapping[int, str]


#: The 11 ONNX model inputs in the exact ``input_names`` order (the V# wire
#: contract).
MODEL_INPUTS: tuple[ModelInput, ...] = (
    ModelInput(ONNX.game_vertex, TORCH.game_vertex, "x", {0: "num_game_vertices"}),
    ModelInput(ONNX.state_vertex, TORCH.state_vertex, "x", {0: "num_state_vertices"}),
    ModelInput(
        ONNX.path_condition_vertex,
        TORCH.path_condition_vertex,
        "x",
        {0: "num_pc_vertices"},
    ),
    ModelInput(
        ONNX.gamevertex_to_gamevertex_index,
        TORCH.gamevertex_to_gamevertex,
        "edge_index",
        {1: "num_gv_gv_edges"},
    ),
    ModelInput(
        ONNX.gamevertex_to_gamevertex_type,
        TORCH.gamevertex_to_gamevertex,
        "edge_type",
        {0: "num_gv_gv_edges"},
    ),
    ModelInput(
        ONNX.gamevertex_history_statevertex_index,
        TORCH.gamevertex_history_statevertex,
        "edge_index",
        {1: "num_gv_history_sv_edges"},
    ),
    # Axis 1 is the fixed per-edge attribute vector
    # [NumOfVisits, StepWhenVisitedLastTime] (see ml/dataset/tensors.py),
    # so only axis 0 is dynamic; torch.export rejects a
    # user-specified dynamic dim that tracing infers as static.
    ModelInput(
        ONNX.gamevertex_history_statevertex_attrs,
        TORCH.gamevertex_history_statevertex,
        "edge_attr",
        {0: "num_gv_history_sv_edges"},
    ),
    ModelInput(
        ONNX.gamevertex_in_statevertex,
        TORCH.gamevertex_in_statevertex,
        "edge_index",
        {1: "num_gv_in_sv_edges"},
    ),
    ModelInput(
        ONNX.statevertex_parentof_statevertex,
        TORCH.statevertex_parentof_statevertex,
        "edge_index",
        {1: "num_sv_parentof_sv_edges"},
    ),
    ModelInput(
        ONNX.pathcondvertex_to_pathcondvertex,
        TORCH.pathcondvertex_to_pathcondvertex,
        "edge_index",
        {1: "num_pc_pc_edges"},
    ),
    ModelInput(
        ONNX.pathcondvertex_to_statevertex,
        TORCH.pathcondvertex_to_statevertex,
        "edge_index",
        {1: "num_pc_sv_edges"},
    ),
)


def load_gamestate(f: t.TextIO) -> GameState:
    return GameState.from_dict(json.load(f))


def create_model_input(
    hetero_data: HeteroData, modifier: t.Callable[[t.Any], t.Any] = lambda x: x
):
    return {
        model_input.name: modifier(
            getattr(hetero_data[model_input.torch_key], model_input.attr)
        )
        for model_input in MODEL_INPUTS
    }


def create_onnxruntime_input(hetero_data: HeteroData):
    return create_model_input(hetero_data, lambda x: x.numpy())


def create_torch_input(hetero_data: HeteroData):
    return tuple(create_model_input(hetero_data).values())


def input_names() -> list[str]:
    """The ONNX input names in model order (derived from ``MODEL_INPUTS``)."""
    return [model_input.name for model_input in MODEL_INPUTS]


def dynamic_axes() -> dict[str, dict[int, str]]:
    """The legacy exporter's ``dynamic_axes`` mapping (derived from
    ``MODEL_INPUTS``)."""
    return {
        model_input.name: dict(model_input.dynamic_axes) for model_input in MODEL_INPUTS
    }


def save_in_onnx(
    model: torch.nn.Module,
    sample_input: HeteroData,
    save_path: pathlib.Path,
    verbose: bool = False,
):
    # With use_segment_matmul unset (None), PyG's RGCNConv selects a kernel
    # via a data-dependent heuristic on the input sizes, which torch.export
    # (the default ONNX exporter backend since torch 2.13) cannot trace. The
    # flag only takes effect when pyg-lib is installed, so disabling it for
    # the export changes no runtime behavior here.
    previous_use_segment_matmul = pyg_backend.use_segment_matmul
    pyg_backend.use_segment_matmul = False
    try:
        with warnings.catch_warnings():
            # Dependency-internal warnings that torch.onnx.export emits for
            # this model; each is safe to suppress (see comments). Scoped to
            # the export call only.
            # PyG RGCNConv's masked_edge_index indexes edge_index with a 1-D
            # boolean mask — exactly the supported case per the warning text.
            warnings.filterwarnings(
                "ignore", category=UserWarning, module=r"torch\.onnx\.symbolic_opset9"
            )
            # PyG MessagePassing._set_size consistency check; the traced
            # value is baked in as a constant, which is correct for a fixed
            # graph.
            warnings.filterwarnings(
                "ignore",
                category=TracerWarning,
                module=r"torch_geometric\.nn\.conv\.message_passing",
            )
            # setup_onnx_logging is @deprecated and called by
            # torch.onnx.export itself on the legacy exporter path —
            # unavoidable from here.
            warnings.filterwarnings(
                "ignore", category=DeprecationWarning, module=r"torch\.onnx\.utils"
            )
            torch.onnx.export(
                model=model,
                args=create_torch_input(sample_input),
                f=save_path,
                verbose=verbose,
                dynamic_axes=dynamic_axes(),
                input_names=input_names(),
                output_names=["out"],
                opset_version=ONNX_OPSET_VERSION,
            )
    finally:
        pyg_backend.use_segment_matmul = previous_use_segment_matmul


def onnx_run(ort_session: onnxruntime.InferenceSession, data: HeteroData) -> ...:
    return ort_session.run(None, create_onnxruntime_input(data))


# The maximum absolute element-wise difference between the torch and the
# ONNX Runtime outputs that verification tolerates. The two runtimes use
# different float32 kernels (matmul orderings, fused ops), so a small
# nonzero difference is legitimate; 1e-4 is tight enough to catch a
# numerically broken conversion while staying above that kernel noise
# (the pre-#597 check effectively allowed 5e-4 by rounding to 3 decimals).
VERIFICATION_ATOL = 1e-4


def verify_outputs(torch_out: torch.Tensor, onnx_out: t.Any) -> None:
    """Verify the ONNX Runtime output against the torch output.

    Compares the full tensors element-wise with an explicit absolute
    tolerance (no rounding, no truncation) and raises ``ValueError`` on
    any divergence beyond it, so a numerically broken conversion cannot
    pass silently (#597).

    Parameters
    ----------
    torch_out:
        The torch model's output for the verification gamestate.
    onnx_out:
        The ONNX Runtime's output (the first element of ``onnx_run``'s
        result).

    Raises
    ------
    ValueError
        If the shapes differ or any element diverges beyond
        ``VERIFICATION_ATOL``.
    """
    onnx_tensor = torch.as_tensor(onnx_out)
    if torch_out.shape != onnx_tensor.shape:
        raise ValueError(
            "ONNX verification failed: shape mismatch "
            f"{tuple(torch_out.shape)} != {tuple(onnx_tensor.shape)}"
        )
    if torch_out.numel() == 0:
        # Both outputs are empty (a degenerate gamestate with no states):
        # there is nothing to diverge.
        return
    diff = (torch_out - onnx_tensor).abs()
    max_diff = diff.max().item()
    if max_diff > VERIFICATION_ATOL:
        raise ValueError(
            "ONNX verification failed: max absolute difference "
            f"{max_diff:.3e} exceeds the tolerance {VERIFICATION_ATOL:.1e} "
            f"(max difference at flattened index {int(diff.argmax())})"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-gamestate",
        dest="sample_gamestate_path",
        type=pathlib.Path,
        required=True,
        help="Path to game state that will initialize ONNX model",
    )
    parser.add_argument(
        "--pytorch-model",
        dest="pytorch_model_path",
        type=pathlib.Path,
        required=True,
        help="Path to torch model weights",
    )
    parser.add_argument(
        "--savepath",
        dest="onnx_savepath",
        type=pathlib.Path,
        required=True,
        help="Path to which ONNX model will be saved",
    )
    parser.add_argument(
        "--model-name",
        dest="model_name",
        type=str,
        required=True,
        help=(
            "registry name of the model to convert "
            "(see ml/models/registry.py), for example: 'NorthernPenguin'"
        ),
    )
    parser.add_argument(
        "--model-kwargs",
        dest="model_kwargs",
        type=pathlib.Path,
        required=True,
        help="Path to .yaml-file with model kwargs",
    )
    parser.add_argument(
        "--verify-on",
        dest="verification_gamestates",
        type=pathlib.Path,
        nargs="*",
        required=False,
        default=[],
        help="Paths to game states to verify against",
    )

    args = parser.parse_args()

    with open(args.model_kwargs, "r") as file:
        model_kwargs = yaml.safe_load(file)

    with open(args.sample_gamestate_path, "r") as gamestate_file:
        sample_gamestate = load_gamestate(gamestate_file)

    veification_gamestates: list[HeteroData] = []
    for gamestate_path in args.verification_gamestates:
        with open(gamestate_path, "r") as gamestate_file:
            veification_gamestates.append(load_gamestate(gamestate_file))

    entrypoint(
        sample_gamestate=sample_gamestate,
        pytorch_model_path=args.pytorch_model_path,
        onnx_savepath=args.onnx_savepath,
        model_def=get_model_class(args.model_name),
        model_kwargs=model_kwargs,
        verification_gamestates=veification_gamestates,
    )


def entrypoint(
    sample_gamestate: GameState,
    pytorch_model_path: pathlib.Path,
    onnx_savepath: pathlib.Path,
    model_def: t.Type[torch.nn.Module],
    model_kwargs: dict[str, t.Any],
    verification_gamestates: list[GameState] = None,
):
    torch_model = model_def(**model_kwargs)
    hetero_sample_gamestate, _ = convert_input_to_tensor(sample_gamestate)
    torch_model.eval()
    # Warm-up forward with the freshly initialized (random) weights: it
    # triggers the model's lazy initialization before the weights are
    # loaded and the graph is traced.
    with torch.no_grad():
        infer(torch_model, hetero_sample_gamestate)
    state_dict: t.OrderedDict = torch.load(pytorch_model_path, map_location="cpu")
    torch_model.load_state_dict(state_dict)

    save_in_onnx(torch_model, hetero_sample_gamestate, onnx_savepath)
    model_onnx = onnx.load(onnx_savepath)
    onnx.checker.check_model(model_onnx)

    if verification_gamestates:
        ort_session = onnxruntime.InferenceSession(onnx_savepath)
        for verification_gamestate in verification_gamestates:
            hetero_verification_gamestate, _ = convert_input_to_tensor(
                verification_gamestate
            )
            with torch.no_grad():
                torch_out = infer(torch_model, hetero_verification_gamestate)
            onnx_out = onnx_run(ort_session, hetero_verification_gamestate)
            verify_outputs(torch_out, onnx_out[0])


if __name__ == "__main__":
    main()
