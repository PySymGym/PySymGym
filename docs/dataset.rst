:description: Expand the training dataset and maintain it with helper tools.
:group: Guides

Dataset
=======

Dataset expansion
-----------------

To enhance the diversity and quality of the training data, the dataset is
extended with additional C# methods sourced from popular open-source algorithm
repositories:

- `TheAlgorithms/C-Sharp <https://github.com/cat923/C-Sharp>`_
- `Stralgo <https://github.com/SaeedGz98/stralgo>`_

Dataset tools
-------------

Two helper scripts under ``tools/dataset_tools/`` automate episode generation
and dataset cleaning. They import the agent package, so run them from that
directory with ``AIAgent`` on ``PYTHONPATH``:

``generate_episodes.py``
    Creates multiple training episodes for each method by varying the
    ``StepsToStart`` parameter, yielding a richer set of examples.

``clean.py``
    Scans training logs and removes episodes where the baseline strategy
    (BFS/DFS) completed execution before the neural network could participate
    (indicated by an "immediate GameOver"). Such episodes are useless for
    training and are discarded.

Both tools rewrite the dataset file atomically: the new content is written
to a temporary file in the same directory and then renamed over the dataset
(``os.replace``, see ``tools/dataset_tools/atomic_io.py``), so a crash or
``KeyboardInterrupt`` mid-write leaves the previous dataset intact instead
of a truncated file.

``clean.py`` detects bad episodes from the free-form training log; the log
phrasing it matches (``immediate GameOver on <MapName>``) is pinned by the
named constants at the top of the module until a structured log source
exists.

Basic usage:

.. code-block:: console

    cd tools/dataset_tools
    PYTHONPATH=../../AIAgent python3 generate_episodes.py
    PYTHONPATH=../../AIAgent python3 clean.py
