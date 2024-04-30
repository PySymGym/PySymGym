import argparse
import importlib
import json
import pathlib
import typing as t
from textwrap import dedent

import onnx
import onnxruntime
import torch
from torch_geometric.data import HeteroData

from common.game import GameState
from ml.inference import ONNX, TORCH
from ml.training.dataset import convert_input_to_tensor

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
    gamestate = sample_input
    torch.onnx.export(
        model=model,
        args=create_torch_input(gamestate),
        f=save_path,
        verbose=verbose,
        dynamic_axes={
            ONNX.game_vertex: [0],
            ONNX.state_vertex: [0],
            ONNX.gamevertex_to_gamevertex_index: [1],
            ONNX.gamevertex_to_gamevertex_type: [0],
            ONNX.gamevertex_history_statevertex_index: [1],
            ONNX.gamevertex_history_statevertex_attrs: [0, 1],
            ONNX.gamevertex_in_statevertex: [1],
            ONNX.statevertex_parentof_statevertex: [1],
        },
        input_names=[
            ONNX.game_vertex,
            ONNX.state_vertex,
            ONNX.gamevertex_to_gamevertex_index,
            ONNX.gamevertex_to_gamevertex_type,
            ONNX.gamevertex_history_statevertex_index,
            ONNX.gamevertex_history_statevertex_attrs,
            ONNX.gamevertex_in_statevertex,
            ONNX.statevertex_parentof_statevertex,
        ],
        output_names=["out"],
        opset_version=ONNX_OPSET_VERSION,
    )


def torch_run(torch_model: torch.nn.Module, data: HeteroData) -> ...:
    torch_in = tuple(create_model_input(data).values())
    return torch_model(*torch_in)


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
        "--import-model-fqn",
        dest="import_model_fqn",
        type=str,
        required=True,
        help=dedent(
            """\n
            model import fully qualified name from AIAgent/ root,
            for example: 'ml.models.TAGSageSimple.model.StateModelEncoder'
            """
        ),
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

    model_kwargs = {
        "hidden_channels": 110,
        "num_of_state_features": 30,
        "num_hops_1": 5,
        "num_hops_2": 4,
        "normalization": True,
    }

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
        model_def=resolve_import_model(args.import_model_fqn),
        model_kwargs=model_kwargs,
        verification_gamestates=veification_gamestates,
    )


def resolve_import_model(import_model_fqn: str) -> t.Type[torch.nn.Module]:
    module, clazz = (
        import_model_fqn[: import_model_fqn.rfind(".")],
        import_model_fqn[import_model_fqn.rfind(".") + 1 :],
    )

    module_def = importlib.import_module(module)
    model_def = getattr(module_def, clazz)

    return model_def


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

    torch_run(torch_model, hetero_sample_gamestate)
    state_dict: t.OrderedDict = torch.load(pytorch_model_path, map_location="cpu")

    torch_model.load_state_dict(state_dict)

    save_in_onnx(torch_model, hetero_sample_gamestate, onnx_savepath)
    model_onnx = onnx.load(onnx_savepath)
    onnx.checker.check_model(model_onnx)

    if verification_gamestates is not []:
        ort_session = onnxruntime.InferenceSession(onnx_savepath)

        for idx, verification_gamestate in enumerate(verification_gamestates, start=1):
            hetero_verification_gamestate, _ = convert_input_to_tensor(
                verification_gamestate
            )
            torch_out = torch_run(torch_model, hetero_verification_gamestate)
            onnx_out = onnx_run(ort_session, hetero_verification_gamestate)

            print(len(verification_gamestate.States))
            print(f"{shorten_output(torch_out)=}")
            print(f"{shorten_output(onnx_out[0])=}")
            print(f"{idx}/{len(verification_gamestates)}")
            assert (
                shorten_output(torch_out) == shorten_output(onnx_out[0])
            ), f"verification failed, {shorten_output(torch_out)} != {shorten_output(onnx_out[0])}"


if __name__ == "__main__":
    main()
