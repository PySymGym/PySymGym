import argparse
import re

import attrs
import pandas as pd


@attrs.define
class PrettyRunResult:
    method: str
    steps: int
    tests: int
    errors: int
    coverage: float


def parse_stats(stats: str) -> tuple[int, int, int, float]:
    steps = re.search("steps: (?P<count>.\d*),", stats).groupdict()["count"]
    coverage = re.search(r"actual %: (?P<number>.\d*\.\d*),", stats).groupdict()[
        "number"
    ]
    tests = re.search("test count: (?P<count>.\d*) ", stats).groupdict()["count"]
    errors = re.search("error count: (?P<count>.\d*)", stats).groupdict()["count"]

    return (int(steps), int(tests), int(errors), float(coverage))


def parse_pretty(data: list[str]) -> pd.DataFrame:
    lines = []
    for line in data:
        line = line.strip("\n")
        if re.match(r"Epoch#", line):
            continue
        if re.fullmatch(r"[+|-]*", line):
            continue
        line_attribites = line.strip().strip("|").split("|")
        line_attribites = [item.strip() for item in line_attribites]
        line_attribites.pop(0)  # pop empty first element and model name
        lines.append(line_attribites)

    rst = []
    for method_full_qual, stats in zip(*lines):
        assert isinstance(method_full_qual, str)
        method_full_qual = method_full_qual.replace("_Method_0", "")
        rst.append(attrs.asdict(PrettyRunResult(method_full_qual, *parse_stats(stats))))

    return pd.DataFrame.from_dict(rst)


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
