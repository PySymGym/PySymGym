PySymGym
========

*Documentation navigation map: start here to find your way around the project.*

``README.md`` is the entrypoint for navigating the project. Every
documentation page opens with a brief one-line description, and this page is
the map: it lists each page with that description, grouped by area. When you
add a page, register it here and give it a brief description -- this keeps
navigation fast for both people and agents.

The published documentation is hosted at
`pysymgym.github.io/PySymGym <https://pysymgym.github.io/PySymGym/>`_.

Documentation map
-----------------

.. list-table::
   :header-rows: 1
   :widths: 14 24 62

   * - Group
     - Page
     - Description
   * - Overview
     - :doc:`architecture`
     - High-level map of the system and where to find things; the starting
       point for navigating the code.
   * - Guides
     - :doc:`usage`
     - Build the engines, run training and tuning, improve the dataset, and
       execute with a trained model.
   * - Guides
     - :doc:`integration`
     - Integrate a new symbolic engine and convert a trained model to ONNX.
   * - Guides
     - :doc:`dataset`
     - Expand the training dataset and maintain it with helper tools.
   * - Guides
     - :doc:`results`
     - Compare the trained AI selector with the best selector of V#.
   * - Development
     - :doc:`developer`
     - Contribution model, tests, style checks, docs build, and CI.
   * - Reference
     - :doc:`reference/index`
     - API reference for the shared, dependency-light helper modules.

.. toctree::
   :hidden:

   architecture
   usage
   integration
   dataset
   results
   developer
   reference/index
