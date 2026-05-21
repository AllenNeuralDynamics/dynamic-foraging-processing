"""Temporary local dataset builder.

This is a local copy of ``aind_behavior_dynamic_foraging.data_contract._dataset.make_dataset``
with the ``BehaviorVideos`` stream removed. It will be removed once the full dataset is acquired.
"""

from pathlib import Path

from aind_behavior_curriculum import TrainerState
from aind_behavior_dynamic_foraging import __semver__
from aind_behavior_dynamic_foraging.rig import AindDynamicForagingRig
from aind_behavior_dynamic_foraging.task_logic import AindDynamicForagingTaskLogic
from aind_behavior_services.session import Session
from contraqctor.contract import Dataset, DataStreamCollection
from contraqctor.contract.harp import (
    DeviceYmlByFile,
    HarpDevice,
)
from contraqctor.contract.json import Json, PydanticModel


def make_dataset(
    root_path: Path,
    name: str = "DynamicForagingDataset",
    description: str = "A Dynamic Foraging dataset",
    version: str = __semver__,
) -> Dataset:
    """Creates a Dataset object for the Dynamic Foraging experiment.

    This function constructs a hierarchical representation of the data streams
    collected during an experiment, including hardware device data, software
    events, and configuration files.

    Parameters
    ----------
    root_path : Path
        Path to the root directory containing the dataset.
    name : str, optional
        Name of the dataset, defaults to ``"DynamicForagingDataset"``.
    description : str, optional
        Description of the dataset, defaults to ``"A Dynamic Foraging dataset"``.
    version : str, optional
        Version of the dataset, defaults to the package version.

    Returns
    -------
    Dataset
        A Dataset object containing a hierarchical representation of all data
        streams from the Dynamic Foraging experiment.
    """
    root_path = Path(root_path)
    return Dataset(
        name=name,
        version=version,
        description=description,
        data_streams=[
            DataStreamCollection(
                name="Behavior",
                description="Data from the Behavior modality",
                data_streams=[
                    Json(
                        name="PreviousMetrics",
                        reader_params=Json.make_params(
                            path=root_path / "behavior/metrics.json",
                        ),
                    ),
                    PydanticModel(
                        name="TrainerState",
                        reader_params=PydanticModel.make_params(
                            model=TrainerState,
                            path=root_path / "behavior/trainer_state.json",
                        ),
                    ),
                    HarpDevice(
                        name="HarpBehavior",
                        reader_params=HarpDevice.make_params(
                            path=root_path / "behavior/Behavior.harp",
                            device_yml_hint=DeviceYmlByFile(),
                        ),
                    ),
                    DataStreamCollection(
                        name="InputSchemas",
                        description="Configuration files for the behavior rig, task_logic and session.",
                        data_streams=[
                            PydanticModel(
                                name="Rig",
                                reader_params=PydanticModel.make_params(
                                    model=AindDynamicForagingRig,
                                    path=root_path / "behavior/Logs/rig_output.json",
                                ),
                            ),
                            PydanticModel(
                                name="TaskLogic",
                                reader_params=PydanticModel.make_params(
                                    model=AindDynamicForagingTaskLogic,
                                    path=root_path / "behavior/Logs/tasklogic_output.json",
                                ),
                            ),
                            PydanticModel(
                                name="Session",
                                reader_params=PydanticModel.make_params(
                                    model=Session,
                                    path=root_path / "behavior/Logs/session_output.json",
                                ),
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def dataset(path, version: str = __semver__) -> Dataset:
    """Build a Dataset at ``path`` using the local (temp) make_dataset."""
    return make_dataset(Path(path), version=version)
