Dataset
=======

*How the training dataset is expanded with extra methods and maintained with
helper tools.*

Dataset expansion
-----------------

To enhance the diversity and quality of the training data, the dataset is
extended with additional C# methods sourced from popular open-source algorithm
repositories:

- `TheAlgorithms/C-Sharp <https://github.com/cat923/C-Sharp.git>`_
- `Stralgo <https://github.com/SaeedGz98/stralgo.git>`_

Dataset tools
-------------

Two helper scripts under ``tools/dataset_tools/`` automate episode generation
and dataset cleaning:

``generate_episodes.py``
    Creates multiple training episodes for each method by varying the
    ``StepsToStart`` parameter, yielding a richer set of examples.

``clean.py``
    Scans training logs and removes episodes where the baseline strategy
    (BFS/DFS) completed execution before the neural network could participate
    (indicated by an "immediate GameOver"). Such episodes are useless for
    training and are discarded.

Basic usage:

.. code-block:: console

    cd tools/dataset_tools
    python3 generate_episodes.py
    python3 clean.py
