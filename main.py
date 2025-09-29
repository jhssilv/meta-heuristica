import argparse

from instance import Instance
from input_parse import parse_args


if __name__ == "__main__":
    args = parse_args()
    instance = Instance(args.input_file)
