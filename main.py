import argparse

from instance import Instance
from input_parse import parse_args
from solver import Solver


if __name__ == "__main__":
    args = parse_args()
    instance = Instance(args.input_file)
    solver = Solver(instance)
    
    print(solver.generate_initial_solution())
    print(solver.is_solution_viable(solver.generate_initial_solution()))
    print(solver.evaluate_solution(solver.generate_initial_solution()))