Usage
=====

*How to build the engines, run training and tuning, improve the dataset, and
execute with a trained model.*

This guide assumes the environment is already set up as described in the
`repository README <https://github.com/PySymGym/PySymGym#install>`__.

Main loop
---------

Hyper-parameters and training artifacts are logged with
`MLflow <https://mlflow.org/>`_. Start its server first:

.. code-block:: console

    cd AIAgent
    poetry run mlflow server -h 127.0.0.1 -p 8080 --serve-artifacts

The training process is a *main loop* that alternates two kinds of runs:
*hyper-parameters tuning* and *dataset improvement with symbolic virtual
machines*. Before the loop, generate an initial dataset. Then alternate tuning
and dataset improvement until the desired result is reached, and finally run
symbolic execution with the trained model.

The point of alternating is mutual improvement: tuning yields the best network
for the current dataset, and that network improves the dataset; a better
dataset then has better hyper-parameters, so tune again.

.. image:: https://raw.githubusercontent.com/PySymGym/PySymGym/main/resources/game_process_illustration.png
   :width: 50%
   :alt: Illustration of the game process

Build symbolic virtual machines and maps
----------------------------------------

To build the symbolic virtual machines (`V# <https://github.com/VSharp-team/VSharp>`_
and `usvm <https://github.com/UnitTestBot/usvm>`_) and the methods used for
training, install .NET 7, cmake, clang, and maven, then run:

.. code-block:: console

    make build_SVMs build_maps

Optionally add new maps under ``maps/`` and integrate another engine (see
`Integrate a new symbolic machine
<https://pysymgym.github.io/PySymGym/integration.html#integrate-a-new-symbolic-machine>`_).

Generate initial dataset
------------------------

Supervised learning needs some initial data, which any path-selection strategy
can produce. This project generates it with one of the strategies from V#:

.. code-block:: console

    make init_data STEPS_TO_SERIALIZE=<MAX_STEPS>

``STEPS_TO_SERIALIZE`` is optional and defaults to 200. The initial dataset is
saved under ``./AIAgent/report/SerializedEpisodes`` and is later updated by the
neural network when it finds a better solution.

Hyper-parameters tuning
-----------------------

Hyper-parameters are tuned with `Optuna <https://optuna.org/>`_. Tuning yields
the best network for the current dataset, which gives a better chance of
improving the dataset in the other kind of run.

1. Create a training configuration. ``./workflow/config_for_tests.yml`` is a
   template. To use the training loss as the tuning objective, set the
   validation config as follows:

   .. code-block:: yaml

       ValidationConfig:
         validation:
           val_type: loss
           batch_size: <DEPENDS_ON_YOUR_RAM_SIZE>

   Configure Optuna, for example:

   .. code-block:: yaml

       OptunaConfig:
         n_startup_trials: 10
         n_trials: 30
         n_jobs: 1
         study_direction: "minimize"

2. Move to the ``AIAgent`` directory and run training:

   .. code-block:: console

       cd AIAgent
       poetry run python3 run_training.py --config path/to/config.yml

Dataset improvement with symbolic virtual machines
---------------------------------------------------

The optimal sequence of symbolic execution steps is unknown, so the best models
from the previous step are used to obtain relatively good ones:

1. Build the SVMs if you have not yet (see `Build symbolic virtual machines
   and maps`_).
2. Using the `workflow example
   <https://github.com/PySymGym/PySymGym/blob/main/workflow/dataset_for_tests_java.json>`_
   as a template, create a configuration with the maps to use, or reuse an
   existing ``maps/*/Maps/dataset.json``.
3. Create a configuration (server and training parameters), again using
   ``./workflow/config_for_tests.yml`` as a template. Add the best weights URI
   and the appropriate trial URI logged by MLflow during tuning:

   .. code-block:: yaml

       weights_uri: mlflow-artifacts:/<EXPERIMENT_ID>/<RUN_ID>/artifacts/<EPOCH>/model.pth

       OptunaConfig:
         ...
         trial_uri: mlflow-artifacts:/<EXPERIMENT_ID>/<RUN_ID>/artifacts/trial.pkl

4. Move to the ``AIAgent`` directory, launch the server manager, and run
   training:

   .. code-block:: console

       cd AIAgent
       poetry run python3 launch_servers.py --config path/to/config.yml
       poetry run python3 run_training.py --config path/to/config.yml

Guide symbolic execution with a trained model
---------------------------------------------

After training, choose the best MLflow-logged model and run symbolic execution:

1. Convert the PyTorch model to ONNX with ``onyx.py`` (see `ONNX conversion
   <https://pysymgym.github.io/PySymGym/integration.html#onnx-conversion>`_).
2. Use the ONNX model to guide execution with your SVM (see `Integrate a new
   symbolic machine
   <https://pysymgym.github.io/PySymGym/integration.html#integrate-a-new-symbolic-machine>`_),
   or use an existing engine extension in this repository:

   - Place the model at
     ``GameServers/VSharp/VSharp.Explorer/models/model.onnx``.
   - Run:

     .. code-block:: console

         dotnet GameServers/VSharp/VSharp.Runner/bin/Release/net8.0/VSharp.Runner.dll \
             --method BinarySearch \
             maps/DotNet/Maps/Root/bin/Release/net8.0/ManuallyCollected.dll \
             --timeout 120 --strat AI --check-coverage
