import argparse
import re

from attrs import asdict, define
import pandas as pd

# The columns of the single-model output; the multi-model output prepends
# a "model" column (see parse_pretty).
SINGLE_MODEL_COLUMNS = ["method", "steps", "tests", "errors", "coverage"]


@define
class PrettyRunResult:
    method: str
    steps: int
    tests: int
    errors: int
    coverage: float


def parse_stats(stats: str) -> tuple[int, int, int, float]:
    steps = re.search(r"steps: (?P<count>\d+),", stats).groupdict()["count"]
    coverage = re.search(r"actual %: (?P<number>\d+\.\d+),", stats).groupdict()[
        "number"
    ]
    tests = re.search(r"test count: (?P<count>\d+) ", stats).groupdict()["count"]
    errors = re.search(r"error count: (?P<count>\d+)", stats).groupdict()["count"]

    return (int(steps), int(tests), int(errors), float(coverage))


def parse_pretty(data: list[str]) -> pd.DataFrame:
    # The first surviving line is the header: an empty corner cell followed
    # by one map-name cell per method column. Every subsequent surviving
    # line is one model row: a model-name cell followed by one stats cell
    # per method column.
    header: list[str] | None = None
    rows: list[tuple[str, list[str]]] = []
    for line in data:
        line = line.strip("\n")
        if re.match(r"Epoch#", line):
            continue
        if re.fullmatch(r"[+|-]*", line):
            continue
        cells = [item.strip() for item in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
        else:
            model, *stats_cells = cells
            rows.append((model, stats_cells))

    if header is None:
        return pd.DataFrame(columns=SINGLE_MODEL_COLUMNS)

    methods = [cell.replace("_Method_0", "") for cell in header[1:]]
    multi_model = len(rows) > 1
    records = []
    for model, stats_cells in rows:
        if len(stats_cells) != len(methods):
            raise ValueError(
                f"model row {model!r} has {len(stats_cells)} stats cells, "
                f"but the header declares {len(methods)} method columns"
            )
        for method, stats in zip(methods, stats_cells, strict=True):
            record = asdict(PrettyRunResult(method, *parse_stats(stats)))
            if multi_model:
                record = {"model": model, **record}
            records.append(record)

    if not records:
        return pd.DataFrame(columns=SINGLE_MODEL_COLUMNS)
    return pd.DataFrame.from_dict(records)


def main():
    parser = argparse.ArgumentParser(
        description="This program parses Common Model results file formatted as a 'pretty' table"
    )
    parser.add_argument(
        "-in",
        "--input",
        type=str,
        required=True,
        help="path to file with 'pretty' table",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="path to save result to",
        default="./parsed.csv",
    )
    args = parser.parse_args()

    with open(args.input, "r") as pretty_table_file:
        data = pretty_table_file.readlines()
    parse_pretty(data).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
