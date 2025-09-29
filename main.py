
from instance import Instance
from input_parse import parse_args
from solver import Solver
import time


if __name__ == "__main__":
    args = parse_args()
    
    max_iter = args.stopping_value
    
    instance = Instance(args.input_file)
    solver = Solver(instance)
    
    initial_value = solver.evaluate_solution(solver.current_solution)
    print(f"Initial solution value: {initial_value}")

    start_time = time.time()
    for i in range(max_iter):
        solver.first_improvement()
        
    final_value = solver.evaluate_solution(solver.current_solution)
    print(f"Final solution value: {final_value}")
    print(f"Done {i+1} iterations in {time.time() - start_time:.2f} seconds.")
    