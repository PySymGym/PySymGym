import argparse
import json
import pathlib
import typing as t
import warnings

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


def load_gamestate(f: t.TextIO) -> GameState:
    return GameState.from_dict(json.load(f))


def create_model_input(
    hetero_data: HeteroData, modifier: t.Callable[[t.Any], t.Any] = lambda x: x
):
    return {
        ONNX.game_vertex: modifier(hetero_data[TORCH.game_vertex].x),
        ONNX.state_vertex: modifier(hetero_data[TORCH.state_vertex].x),
        ONNX.path_condition_vertex: modifier(
            hetero_data[TORCH.path_condition_vertex].x
        ),
        ONNX.gamevertex_to_gamevertex_index: modifier(
            hetero_data[TORCH.gamevertex_to_gamevertex].edge_index
        ),
        ONNX.gamevertex_to_gamevertex_type: modifier(
            hetero_data[TORCH.gamevertex_to_gamevertex].edge_type
        ),
        ONNX.gamevertex_history_statevertex_index: modifier(
            hetero_data[TORCH.gamevertex_history_statevertex].edge_index
        ),
        ONNX.gamevertex_history_statevertex_attrs: modifier(
            hetero_data[TORCH.gamevertex_history_statevertex].edge_attr
        ),
        ONNX.gamevertex_in_statevertex: modifier(
            hetero_data[TORCH.gamevertex_in_statevertex].edge_index
        ),
        ONNX.statevertex_parentof_statevertex: modifier(
            hetero_data[TORCH.statevertex_parentof_statevertex].edge_index
        ),
        ONNX.pathcondvertex_to_pathcondvertex: modifier(
            hetero_data[TORCH.pathcondvertex_to_pathcondvertex].edge_index
        ),
        ONNX.pathcondvertex_to_statevertex: modifier(
            hetero_data[TORCH.pathcondvertex_to_statevertex].edge_index
        ),
    }


def create_onnxruntime_input(hetero_data: HeteroData):
    return create_model_input(hetero_data, lambda x: x.numpy())


def create_torch_input(hetero_data: HeteroData):
    return tuple(create_model_input(hetero_data).values())


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
                # Axis names are required: torch.onnx.utils._validate_dynamic_axes
                # warns once per input whose value is an axis list ("No names were
                # found for specified dynamic axes") and auto-generates the dim
                # params instead. Names must be unique per distinct size — ONNX
                # treats equal dim_param strings as the same symbolic dimension —
                # except that an edge type's index (axis 1) and its per-edge
                # attributes (axis 0) always hold the same number of edges, so
                # they share a name.
                dynamic_axes={
                    ONNX.game_vertex: {0: "num_game_vertices"},
                    ONNX.state_vertex: {0: "num_state_vertices"},
                    ONNX.path_condition_vertex: {0: "num_pc_vertices"},
                    ONNX.gamevertex_to_gamevertex_index: {1: "num_gv_gv_edges"},
                    ONNX.gamevertex_to_gamevertex_type: {0: "num_gv_gv_edges"},
                    ONNX.gamevertex_history_statevertex_index: {
                        1: "num_gv_history_sv_edges"
                    },
                    # Axis 1 is the fixed per-edge attribute vector
                    # [NumOfVisits, StepWhenVisitedLastTime] (see ml/dataset.py),
                    # so only axis 0 is dynamic; torch.export rejects a
                    # user-specified dynamic dim that tracing infers as static.
                    ONNX.gamevertex_history_statevertex_attrs: {
                        0: "num_gv_history_sv_edges"
                    },
                    ONNX.gamevertex_in_statevertex: {1: "num_gv_in_sv_edges"},
                    ONNX.statevertex_parentof_statevertex: {
                        1: "num_sv_parentof_sv_edges"
                    },
                    ONNX.pathcondvertex_to_pathcondvertex: {1: "num_pc_pc_edges"},
                    ONNX.pathcondvertex_to_statevertex: {1: "num_pc_sv_edges"},
                },
                input_names=[
                    ONNX.game_vertex,
                    ONNX.state_vertex,
                    ONNX.path_condition_vertex,
                    ONNX.gamevertex_to_gamevertex_index,
                    ONNX.gamevertex_to_gamevertex_type,
                    ONNX.gamevertex_history_statevertex_index,
                    ONNX.gamevertex_history_statevertex_attrs,
                    ONNX.gamevertex_in_statevertex,
                    ONNX.statevertex_parentof_statevertex,
                    ONNX.pathcondvertex_to_pathcondvertex,
                    ONNX.pathcondvertex_to_statevertex,
                ],
                output_names=["out"],
                opset_version=ONNX_OPSET_VERSION,
            )
    finally:
        pyg_backend.use_segment_matmul = previous_use_segment_matmul


def onnx_run(ort_session: onnxruntime.InferenceSession, data: HeteroData) -> ...:
    return ort_session.run(None, create_onnxruntime_input(data))


def shorten_output(torch_out, round_to: int = 3):
    shortened = list(map(lambda x: round(x, round_to), torch_out.flatten().tolist()))
    if len(shortened) > 10:
        shortened = shortened[:10] + ["..."]
    return shortened


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
    infer(torch_model, hetero_sample_gamestate)
    state_dict: t.OrderedDict = torch.load(pytorch_model_path, map_location="cpu")
    torch_model.load_state_dict(state_dict)

    save_in_onnx(torch_model, hetero_sample_gamestate, onnx_savepath)
    model_onnx = onnx.load(onnx_savepath)
    onnx.checker.check_model(model_onnx)

    if verification_gamestates:
        ort_session = onnxruntime.InferenceSession(onnx_savepath)
        for idx, verification_gamestate in enumerate(verification_gamestates, start=1):
            hetero_verification_gamestate, _ = convert_input_to_tensor(
                verification_gamestate
            )
            torch_out = infer(torch_model, hetero_verification_gamestate)
            onnx_out = onnx_run(ort_session, hetero_verification_gamestate)

            print(len(verification_gamestate.States))
            print(f"{shorten_output(torch_out)=}")
            print(f"{shorten_output(onnx_out[0])=}")
            print(f"{idx}/{len(verification_gamestates)}")
            assert shorten_output(torch_out) == shorten_output(onnx_out[0]), (
                f"verification failed, {shorten_output(torch_out)} != {shorten_output(onnx_out[0])}"
            )


if __name__ == "__main__":
    main()
