:description: Build the engines, run training and tuning, improve the dataset, and execute with a trained model.
:group: Guides

Usage
=====

This guide assumes the environment is already set up as described in the
:doc:`setup` page.

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
training, set up the toolchain as described in the :doc:`setup` page, then run:

.. code-block:: console

    make build_SVMs build_maps

Optionally add new maps under ``maps/`` and integrate another engine (see
:ref:`integrate-a-new-symbolic-machine`).

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

   The ``TrainingConfig`` section requires a ``seed``. It makes the run
   reproducible: the dataset split, step sampling, validation map order, and
   model weight initialization are seeded with it, each trial re-seeds from a
   seed derived from (study seed, trial number), and the Optuna sampler is
   seeded with the study seed — so two runs of the same configuration produce
   identical splits, samples, and hyper-parameter suggestions.

   Each trial writes its artifacts to ``report/trials/<trial_number>/``
   (model weights, model kwargs, and the SVM statistics table for SVM
   validation), so parallel trials cannot overwrite each other; the study
   snapshots ``study.pkl`` and ``trial.pkl`` stay at the top of ``report/``.

   Each trial also runs in its own MLflow run named after the trial number.
   After each trial finishes, the ``study.pkl`` and ``trial.pkl`` snapshots
   are logged to that run together with a ``best_trial_number`` tag holding
   the best trial among those completed so far;
   ``derive_dataset_improvement_config.py`` locates the best trial's
   artifacts through that tag.

   A trial stops early when its validation result stops improving. On early
   stop (or after the final epoch), the best checkpoint — the epoch with the
   best validation result — is restored into the model and exported, so the
   retained ``model.pth`` and the highest-epoch ``<epoch>/model.pth`` MLflow
   artifact hold the best weights, and the trial's value is that best
   result, not the last epoch's.

2. Move to the ``AIAgent`` directory and run training:

   .. code-block:: console

       cd AIAgent
       poetry run python3 run_training.py --config path/to/config.yml

   The entrypoints are CWD-independent: the artifacts always land under
   ``AIAgent/`` and the config's relative paths resolve against the config
   file's own directory, so the command works from any directory (the
   ``cd`` only locates the script).

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
   ``./workflow/config_for_tests.yml`` as a template. The improvement run must
   start from the tuning run's best model, so the config needs two URIs: the
   best weights and the trial that produced them. Derive both from MLflow
   instead of copying them by hand — the script reads the server and experiment
   from the config's ``MLFlowConfig`` section and writes the config plus:

   .. code-block:: console

       cd AIAgent
       poetry run python3 derive_dataset_improvement_config.py \
           --base-config path/to/config.yml --output-config path/to/derived.yml

   .. code-block:: yaml

       weights_uri: mlflow-artifacts:/<EXPERIMENT_ID>/<RUN_ID>/artifacts/<EPOCH>/model.pth

       OptunaConfig:
         ...
         trial_uri: mlflow-artifacts:/<EXPERIMENT_ID>/<RUN_ID>/artifacts/trial.pkl

   The derived config's path values are absolute (anchored against the base
   config's directory), so it loads from any directory.

4. Move to the ``AIAgent`` directory, launch the server manager, and run
   training with the derived config:

   .. code-block:: console

       cd AIAgent
       poetry run python3 launch_servers.py --config path/to/derived.yml
       poetry run python3 run_training.py --config path/to/derived.yml

   The improvement run replays the frozen trial through a fresh study,
   enqueuing its hyper-parameters once per replay. It persists
   ``study.pkl``/``trial.pkl`` and the per-trial MLflow artifacts exactly
   like a tuning run, so the next round of the loop derives its config the
   same way.

SVM validation failure flag
---------------------------

The SVM validation section of the config (``val_type: svms_model`` or
``svms_each_step``) has an opt-in ``fail_immediately`` flag, off by default.
When enabled:

- a game that ends before all planned steps are played without reaching 100%
  coverage is marked as a failed map — the symbolic engine stopping that early
  is an engine defect, not a model quality issue; by default it only produces
  a warning; and
- any failed map (timeout, engine error, ...) fails the run with a non-zero
  exit.

.. _guide-symbolic-execution:

Guide symbolic execution with a trained model
---------------------------------------------

After training, choose the best MLflow-logged model and run symbolic execution:

1. Convert the PyTorch model to ONNX with ``onyx.py`` (see
   :ref:`onnx-conversion`).
2. Use the ONNX model to guide execution with your SVM (see
   :ref:`integrate-a-new-symbolic-machine`),
   or use an existing engine extension in this repository:

   - Place the model at
     ``GameServers/VSharp/VSharp.Explorer/models/model.onnx``.
   - Run:

     .. code-block:: console

         dotnet GameServers/VSharp/VSharp.Runner/bin/Release/net8.0/VSharp.Runner.dll \
             --method BinarySearch \
             maps/DotNet/Maps/Root/bin/Release/net8.0/ManuallyCollected.dll \
             --timeout 120 --strat AI --check-coverage
