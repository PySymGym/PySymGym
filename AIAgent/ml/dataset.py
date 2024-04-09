import json
import multiprocessing as mp
import os
from functools import partial
from pathlib import Path
from typing import Dict, TypeAlias

import numpy as np
import torch
import tqdm
from common.game import GameState
from ml.data_loader_compact import MapStatistics, ServerDataloaderHeteroVector, Step

GAMESTATESUFFIX = "_gameState"
STATESUFFIX = "_statesInfo"
MOVEDSTATESUFFIX = "_movedState"

StateId: TypeAlias = int
StateIndex: TypeAlias = int
StateMap: TypeAlias = Dict[StateId, StateIndex]

VertexId: TypeAlias = int
VertexIndex: TypeAlias = int
VertexMap: TypeAlias = Dict[VertexId, VertexIndex]

MapName: TypeAlias = str
CoveragePercent: TypeAlias = int
TestsNumber: TypeAlias = int
ErrorsNumber: TypeAlias = int
StepsNumber: TypeAlias = int
Result: TypeAlias = tuple[CoveragePercent, TestsNumber, ErrorsNumber, StepsNumber]


class Dataset:
    def __init__(
        self,
        raw_dir: Path,
        processed_dir: Path,
        num_processes: int = mp.cpu_count() - 1,
    ):
        self.num_processes = num_processes
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir

        if len(os.listdir(processed_dir)) == 0:
            self.process()

    def process_step(self, map_path: Path, step_id: str) -> Step:
        def get_states_properties(
            file_path: str, state_map: StateMap
        ) -> Step.StatesProperties:
            """Get tensor for states"""
            expected = dict()
            f = open(
                file_path
            )  # without resource manager in order to escape file descriptors leaks
            data = json.load(f)
            f.close()
            state_set = set()
            for d in data:
                sid = d["StateId"]
                if sid not in state_set:
                    state_set.add(sid)
                    values = [
                        d["NextInstructionIsUncoveredInZone"],
                        d["ChildNumberNormalized"],
                        d["VisitedVerticesInZoneNormalized"],
                        d["Productivity"],
                        d["DistanceToReturnNormalized"],
                        d["DistanceToUncoveredNormalized"],
                        d["DistanceToNotVisitedNormalized"],
                        d["ExpectedWeight"],
                    ]
                    expected[sid] = np.array(values)
            ordered = []
            ordered_by_index = list(
                zip(*sorted(state_map.items(), key=lambda x: x[1]))
            )[0]
            for k in ordered_by_index:
                ordered.append(expected[k])
            return torch.tensor(np.array(ordered), dtype=torch.float)

        def get_states_distribution(
            file_path: str, state_map: StateMap
        ) -> Step.StatesDistribution:
            states_distribution = torch.zeros([len(state_map.keys()), 1])
            f = open(file_path)
            state_id = int(f.read())
            f.close()
            states_distribution[state_map[state_id]] = 1
            return states_distribution

        f = open(os.path.join(map_path, step_id + GAMESTATESUFFIX))
        data = json.load(f)
        f.close()
        graph, state_map = ServerDataloaderHeteroVector.convert_input_to_tensor(
            GameState.from_dict(data)
        )
        if graph is not None:
            # add_expected values
            states_properties = get_states_properties(
                os.path.join(map_path, step_id + STATESUFFIX),
                state_map,
            )
        distribution = get_states_distribution(
            os.path.join(map_path, step_id + MOVEDSTATESUFFIX), state_map
        )
        step = Step(graph, distribution, states_properties)
        return step

    def load_dataset(self) -> list[MapStatistics]:
        def get_result(map_name: MapName):
            f = open(os.path.join(self.raw_dir, map_name, "result"))
            result = f.read()
            f.close()
            result = tuple(map(lambda x: int(x), result.split()))
            return (result[0], -result[1], -result[2], result[3])

        def get_step_ids(map_name: str):
            file_names = os.listdir(os.path.join(self.raw_dir, map_name))
            step_ids = list(
                set(map(lambda file_name: file_name.split("_")[0], file_names))
            )
            step_ids.remove("result")
            return step_ids

        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     for map_name in tqdm.tqdm(
        #         os.listdir(self.raw_dir), desc="Dataset processing"
        #     ):
        #         map_path = os.path.join(self.raw_dir, map_name)
        #         process_steps_task = partial(self.process_step, map_path)
        #         ids = get_step_ids(map_name)

        #         #steps = executor.map(process_steps_task, ids)
        #         futures = [executor.submit(process_steps_task, f) for f in ids]
        #         steps = [fut.result() for fut in futures]
        #         #steps = as_completed(futures)
        #         yield MapStatistics(
        #             MapName=map_name,
        #             Steps=steps,
        #             Result=get_result(map_name),
        #         )
        maps_data = []
        with mp.Pool(self.num_processes) as p:
            for map_name in tqdm.tqdm(
                os.listdir(self.raw_dir), desc="Dataset processing"
            ):
                map_path = os.path.join(self.raw_dir, map_name)
                process_steps_task = partial(self.process_step, map_path)
                ids = get_step_ids(map_name)
                steps = list(map(process_steps_task, sorted(ids)))
                maps_data.append(
                    MapStatistics(
                        MapName=map_name,
                        Steps=steps,
                        Result=get_result(map_name),
                    )
                )
        return maps_data

    def process(self):
        raise NotImplementedError
