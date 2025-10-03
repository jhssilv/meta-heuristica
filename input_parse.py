import argparse

# python main.py instances/01.txt 1000 42

def parse_args():
    parser = argparse.ArgumentParser(description="Heuristic parameters")
    parser.add_argument("input_file", type=str,
                        help="Full instance file path")
    parser.add_argument("stopping_value", type=int,
                        help="Max iterations")
    parser.add_argument("seed_or_param", type=int,
                        help="Random seed")
    parser.add_argument("--simplified", action="store_true",
                        help="Show simplified output")
    return parser.parse_args()
