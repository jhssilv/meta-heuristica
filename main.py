import random
import numpy as np
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

    solver.run(max_iter)
    
    print(f"Final solution is viable: {solver.is_solution_viable(solver.best_solution)}")