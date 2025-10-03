import random
import numpy as np
import time
from instance import Instance
from input_parse import parse_args
from solver import Solver

if __name__ == "__main__":
    args = parse_args()
    
    max_iter = args.stopping_value
    seed = args.seed_or_param
    simplified = args.simplified

    random.seed(seed)
    np.random.seed(seed)

    instance = Instance(args.input_file)
    solver = Solver(instance, seed=seed, show_simplified_output=simplified)
    
    print(f"Using seed: {seed}")

    initial_value = solver.evaluate_solution(solver.current_solution)
    print(f"Initial solution value: {initial_value}")
    print(f"Initial solution is viable: {solver.is_solution_viable(solver.current_solution)}")

    start_time = time.perf_counter()

    solver.run(max_iter)

    end_time = time.perf_counter()

    print(f"\nSolver execution time: {end_time - start_time:.2f} seconds")
    print(f"Final solution is viable: {solver.is_solution_viable(solver.best_solution)}")