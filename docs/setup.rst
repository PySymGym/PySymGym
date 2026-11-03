:description: Install the toolchain, configure the Python environment, and keep submodule checkouts healthy on a fresh machine.
:group: Development

Setup
=====

This page is the single source of truth for development-environment setup:
toolchain requirements, the Python environment, per-component build
prerequisites, and submodule health. The :doc:`usage` guide assumes everything
here is already in place.

Toolchain
---------

.. list-table::
   :header-rows: 1

   * - Component
     - Requirement
     - Source of truth
   * - .NET SDK
     - 8.x (CI also installs 7 and 9)
     - ``.github/docker/Dockerfile``, ``GameServers/VSharp/global.json``
   * - JDK
     - 11–21; set ``JAVA_HOME`` when several JDKs are installed
     - ``GameServers/usvm/gradle/wrapper/gradle-wrapper.properties`` (Gradle 8.6 runs on Java ≤21), ``.github/workflows/build_and_test_usvm.yaml`` (pins 11)
   * - cmake, clang
     - any recent version
     - ``.github/docker/Dockerfile``
   * - Maven
     - any recent version
     - ``Makefile`` (``build_LogicNG``, ``build_jflex``; local development only — CI never builds the Java maps)
   * - Python
     - 3.14.x
     - ``pyproject.toml`` (``~3.14``); CI runs 3.14.4

Python version decision
-----------------------

The evaluation of moving off Python 3.11 (issue #643, part of hub #641)
adopted **Python 3.14** as the target and rejected 3.12/3.13: 3.14 removes
the suppressed W11 warning (numpy's stray ``.pyc`` is cp311-specific) exactly
as 3.12/3.13 would, and it is the newest stable interpreter in scope, so the
modernization happens in a single re-lock rather than an intermediate
3.12/3.13 re-lock with no unique benefit. The GPU blocker that originally
held 3.14 back is void — the #662 probe (``AIAgent/probe_gpu_torch.py``)
degrades the e2e pipeline to CPU when the locked torch cannot run on the
visible GPU (both GPUs are sm_61, unsupported by torch ≥2.9). Python 3.15 is
out of scope for this hub. The migration (interpreter bump, torch bump to
2.15, full re-lock, workflow pin changes) is tracked in #644; the full
evaluation and rationale are recorded on #643.

Python environment
------------------

Install a Python in the supported range (uv is recommended because it does not
touch the system interpreter), then point Poetry at it:

.. code-block:: console

    pipx install "poetry>=2.3.0"
    pipx install uv
    uv python install 3.14
    poetry env use $(uv python find 3.14)
    poetry install

If the distribution's default ``python3`` is outside the supported range (for
example Ubuntu 24.04 ships 3.12), skipping ``poetry env use`` makes Poetry pick
it and the install fails: the project supports 3.14.x only.

Per-component build prerequisites
---------------------------------

- **VSharp.** .NET SDK 8.x: ``GameServers/VSharp/global.json`` pins SDK 8.0.0.
  Build with ``make build_VSharp``.
- **usvm.** A JDK in the supported range; when several are installed, point
  ``JAVA_HOME`` at the one to use (for example
  ``/usr/lib/jvm/java-17-openjdk-amd64``). Build with ``make build_usvm``.
- **Maps.** cmake and clang (used by VSharp.CoverageInstrumenter). Build with
  ``make build_maps``.
- **runstrat tests.** Two pre-built net8.0 artifacts that no other target
  produces:

  .. code-block:: console

      dotnet build tools/runstrat/resources/ForTests -c Release
      dotnet build GameServers/VSharp/VSharp.Runner -c Release

  Note that ``make build_SVMs`` builds ``VSharp.ML.GameServer.Runner``, not
  ``VSharp.Runner``.

Submodules
----------

Initialize all submodules after cloning:

.. code-block:: console

    git submodule update --init --recursive

Check their health (reports uninitialized or wiped checkouts and exits
non-zero on any finding):

.. code-block:: console

    make check_submodules

A wiped worktree (files deleted inside a submodule) is repaired with:

.. code-block:: console

    git -C <submodule-path> reset --hard HEAD

If the submodule's ``.git`` directory is missing, re-run
``git submodule update --init --recursive`` instead.

GPU
---

To use a GPU, the correct ``torch`` and ``torch_geometric`` versions should be
installed depending on your host device. You may first need to ``pip uninstall``
these packages, provided by requirements. Then follow installation instructions
provided on the `torch <https://pytorch.org/get-started/locally/>`_ and
`torch_geometric <https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html#installation-from-wheels>`_
websites.
