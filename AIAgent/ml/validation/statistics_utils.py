import numpy as np


def avg_by_attr(results, path_to_coverage: str) -> int:
    if not results:
        return -1
    coverage = np.average([getattr(result, path_to_coverage) for result in results])
    return coverage
