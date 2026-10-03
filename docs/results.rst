:description: Compare the trained AI selector with the best selector of V#.
:group: Guides

Results
=======

The selector based on the model (AI) is compared with the best selector of V#.
All metrics except coverage are reported for methods that reach equal coverage.

+------------------------------------------+------------------------------------------+
| |coverage|                               | |tests|                                  |
+------------------------------------------+------------------------------------------+
| |time|                                   | |errors|                                 |
+------------------------------------------+------------------------------------------+

.. |coverage| image:: https://raw.githubusercontent.com/PySymGym/PySymGym/main/resources/coverage.svg
   :width: 100%

.. |tests| image:: https://raw.githubusercontent.com/PySymGym/PySymGym/main/resources/tests_eq_coverage.svg
   :width: 100%

.. |time| image:: https://raw.githubusercontent.com/PySymGym/PySymGym/main/resources/total_time_eq_coverage.svg
   :width: 100%

.. |errors| image:: https://raw.githubusercontent.com/PySymGym/PySymGym/main/resources/errors_eq_coverage.svg
   :width: 100%

*ExecutionTreeContributedCoverage* is claimed to be the best searcher in V#
for test generation and was chosen as the reference. Both searchers were
executed with a timeout of 180 seconds for each method.

The model demonstrates slightly better average coverage (87.97% vs 87.61%) in
slightly worse average time (22.8 s vs 18.5 s). Detailed analysis shows that the
trained model generates significantly fewer tests (as expected with respect to
the objective function) but reports fewer potential errors (which also
correlates with the objective function).