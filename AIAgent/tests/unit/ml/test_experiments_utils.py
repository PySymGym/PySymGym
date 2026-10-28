"""Unit tests for the helpers in ``ml.training.experiments_utils``."""

from pathlib import Path

import numpy as np
import pytest
import torch
from common.game import GameMap
from ml.inference import TORCH
from ml.models.NorthernPenguin.model import StateModelEncoder as NorthernPenguinEncoder
from ml.training.experiments_utils import (
    euclidean_dist,
    find_dominators_in_cfg,
    find_entry_points,
    find_unfinished_maps,
    get_model,
    remove_call_return_edges,
    sample_dataset,
    sync_dataset_with_description,
)
from torch_geometric.data import Data, HeteroData

pytestmark = pytest.mark.unit

# ``get_model`` overwrites ``lin_last`` with random weights of shape
# ``[1, 8]`` / ``[1]``, so the model must end in ``Linear(8, 1)``.
GET_MODEL_KWARGS = {
    "hidden_channels": 8,
    "num_of_state_features": 8,
    "num_hops_1": 1,
    "num_hops_2": 1,
    "normalization": True,
    "num_pc_layers": 1,
}


def test_euclidean_dist_is_zero_for_a_single_point():
    assert euclidean_dist(torch.tensor([[1.0]]), torch.tensor([[0.0]])) == 0


def test_euclidean_dist_min_shifts_both_tensors():
    y_pred = torch.tensor([[0.0], [2.0]])
    y_true = torch.tensor([[0.0], [0.0]])

    assert torch.isclose(euclidean_dist(y_pred, y_true), torch.tensor(2.0))


def test_euclidean_dist_is_translation_invariant():
    y_pred = torch.tensor([[0.0], [3.0]])
    y_true = torch.tensor([[1.0], [2.0]])
    shift = 5.0

    assert torch.isclose(
        euclidean_dist(y_pred + shift, y_true + shift),
        euclidean_dist(y_pred, y_true),
    )


def test_remove_call_return_edges_drops_types_one_and_two():
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    cfg = Data(
        x=torch.zeros(4, 1),
        edge_index=edge_index,
        edge_attr=torch.tensor([0, 1, 2, 0]),
    )

    result = remove_call_return_edges(cfg)

    assert torch.equal(result.edge_index, torch.tensor([[0, 3], [1, 0]]))


def test_remove_call_return_edges_keeps_everything_without_call_edges():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    cfg = Data(
        x=torch.zeros(2, 1),
        edge_index=edge_index.clone(),
        edge_attr=torch.tensor([0, 0]),
    )

    result = remove_call_return_edges(cfg)

    assert torch.equal(result.edge_index, edge_index)


def test_remove_call_return_edges_handles_empty_graph():
    cfg = Data(
        x=torch.zeros(2, 1),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0,), dtype=torch.long),
    )

    result = remove_call_return_edges(cfg)

    assert result.edge_index.shape == (2, 0)


def test_find_entry_points_returns_vertices_that_are_never_a_target():
    cfg = Data(
        x=torch.zeros(4, 1),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
    )

    assert find_entry_points(cfg) == [0, 3]


def test_find_entry_points_returns_all_vertices_for_an_empty_graph():
    cfg = Data(
        x=torch.zeros(3, 1),
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )

    assert find_entry_points(cfg) == [0, 1, 2]


def test_get_model_reloads_the_weights_overwriting_lin_last(tmp_path: Path) -> None:
    def model_init() -> NorthernPenguinEncoder:
        return NorthernPenguinEncoder(**GET_MODEL_KWARGS)

    original = model_init()
    weights_path = tmp_path / "weights.pt"
    torch.save(original.state_dict(), weights_path)

    # Predict the exact random overwrite before ``get_model`` draws it.
    rng_state = np.random.get_state()
    expected_weight = np.random.random([1, 8])
    expected_bias = np.random.random([1])
    np.random.set_state(rng_state)

    model = get_model(weights_path, model_init)

    expected = dict(original.state_dict())
    expected["lin_last.weight"] = torch.tensor(expected_weight, dtype=torch.float32)
    expected["lin_last.bias"] = torch.tensor(expected_bias, dtype=torch.float32)
    for name, param in expected.items():
        assert torch.equal(model.state_dict()[name], param), name


def test_find_unfinished_maps_prints_started_but_not_finished_maps(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_file = tmp_path / "server.log"
    log_file.write_text(
        "Start 1 map_A 2 8080\nFinish 1 map_A 2 8080\nStart 1 map_B 2 8080\n"
    )

    find_unfinished_maps(log_file)

    assert capsys.readouterr().out == "map_B\n"


def test_find_unfinished_maps_prints_nothing_for_a_finish_without_start(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_file = tmp_path / "server.log"
    log_file.write_text("Finish 1 map_C 2 8080\n")

    find_unfinished_maps(log_file)

    assert capsys.readouterr().out == ""


def test_find_unfinished_maps_tracks_ports_independently(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_file = tmp_path / "server.log"
    log_file.write_text(
        "Start 1 map_A 2 8080\nStart 1 map_B 2 8081\nFinish 1 map_A 2 8080\n"
    )

    find_unfinished_maps(log_file)

    assert capsys.readouterr().out == "map_B\n"


def test_sync_dataset_with_description_removes_maps_missing_from_the_description(
    tmp_path: Path, game_map_factory
) -> None:
    dataset_path = tmp_path / "dataset"
    for map_name in ("mapA", "mapB", "mapC"):
        (dataset_path / map_name).mkdir(parents=True)
        (dataset_path / map_name / "step.pt").write_bytes(b"data")
    description_path = tmp_path / "description.json"
    description_path.write_text(
        GameMap.schema().dumps(
            [game_map_factory("mapA"), game_map_factory("mapB")], many=True
        )
    )

    sync_dataset_with_description(dataset_path, description_path)

    assert not (dataset_path / "mapC").exists()
    assert (dataset_path / "mapA" / "step.pt").exists()
    assert (dataset_path / "mapB" / "step.pt").exists()


def _cfg_hetero(num_vertices: int, edge_index: torch.Tensor, edge_type: torch.Tensor):
    graph = HeteroData()
    graph[TORCH.game_vertex].x = torch.zeros(num_vertices, 7)
    graph[TORCH.gamevertex_to_gamevertex].edge_index = edge_index
    graph[TORCH.gamevertex_to_gamevertex].edge_type = edge_type
    return graph


def test_find_dominators_in_cfg_appends_dominator_edges_from_each_entry_point() -> None:
    # Entry point 0 dominates 1, 2 and 3; the appended edges run from the
    # dominator to the dominated vertex with edge type 3.
    graph = _cfg_hetero(
        4,
        torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]]),
        torch.tensor([0, 0, 0, 0]),
    )

    find_dominators_in_cfg(graph)

    assert torch.equal(
        graph[TORCH.gamevertex_to_gamevertex].edge_index,
        torch.tensor([[0, 0, 1, 2, 0, 0, 0], [1, 2, 3, 3, 2, 1, 3]]),
    )
    assert torch.equal(
        graph[TORCH.gamevertex_to_gamevertex].edge_type,
        torch.tensor([0, 0, 0, 0, 3, 3, 3]),
    )


def test_find_dominators_in_cfg_handles_multiple_entry_points() -> None:
    # Vertices 0 and 1 are both entry points; each dominates vertex 2.
    graph = _cfg_hetero(
        3,
        torch.tensor([[0, 1], [2, 2]]),
        torch.tensor([0, 0]),
    )

    find_dominators_in_cfg(graph)

    assert torch.equal(
        graph[TORCH.gamevertex_to_gamevertex].edge_index,
        torch.tensor([[0, 1, 0, 1], [2, 2, 2, 2]]),
    )
    assert torch.equal(
        graph[TORCH.gamevertex_to_gamevertex].edge_type,
        torch.tensor([0, 0, 3, 3]),
    )


def test_find_dominators_in_cfg_raises_without_an_entry_point() -> None:
    graph = _cfg_hetero(
        2,
        torch.tensor([[0, 1], [1, 0]]),
        torch.tensor([0, 0]),
    )

    with pytest.raises(ValueError, match="There is no entry points to find dominators"):
        find_dominators_in_cfg(graph)


def _make_sample_layout(root: Path) -> None:
    for map_name, files in (("mapA", ("a.pt", "b.pt")), ("mapB", ("c.pt",))):
        map_dir = root / "dataset" / map_name
        map_dir.mkdir(parents=True)
        for file_name in files:
            torch.save(torch.zeros(1), map_dir / file_name)
        (map_dir / "result").write_text("result")


def test_sample_dataset_copies_every_file_at_full_percentage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The function unpacks ``normpath(path).split(os.sep)`` into exactly four
    # parts, so the dataset must sit two levels below the working directory.
    _make_sample_layout(tmp_path / "root")
    monkeypatch.chdir(tmp_path)

    sample_dataset(Path("root/dataset"), Path("root/save"), 1.0)

    for map_name, files in (("mapA", ("a.pt", "b.pt")), ("mapB", ("c.pt",))):
        for file_name in files:
            assert (tmp_path / "root" / "save" / map_name / file_name).exists()
        assert (tmp_path / "root" / "save" / map_name / "result").exists()


def test_sample_dataset_copies_the_rounded_percentage_and_the_map_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_sample_layout(tmp_path / "root")
    monkeypatch.chdir(tmp_path)

    sample_dataset(Path("root/dataset"), Path("root/save"), 0.34)

    saved_maps = sorted(p.name for p in (tmp_path / "root" / "save").iterdir())
    assert len(saved_maps) == 1
    map_name = saved_maps[0]
    assert len(list((tmp_path / "root" / "save" / map_name).glob("*.pt"))) == 1
    assert (tmp_path / "root" / "save" / map_name / "result").exists()


def test_sample_dataset_into_a_populated_save_dir_skips_making_the_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_sample_layout(tmp_path / "root")
    monkeypatch.chdir(tmp_path)

    sample_dataset(Path("root/dataset"), Path("root/save"), 1.0)
    sample_dataset(Path("root/dataset"), Path("root/save"), 1.0)

    for map_name, files in (("mapA", ("a.pt", "b.pt")), ("mapB", ("c.pt",))):
        for file_name in files:
            assert (tmp_path / "root" / "save" / map_name / file_name).exists()
        assert (tmp_path / "root" / "save" / map_name / "result").exists()
