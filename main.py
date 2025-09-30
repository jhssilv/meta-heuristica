
from instance import Instance
from input_parse import parse_args
from solver import Solver


if __name__ == "__main__":
    args = parse_args()
    
    max_iter = args.stopping_value
    
    instance = Instance(args.input_file)
    solver = Solver(instance)
    
    initial_value = solver.evaluate_solution(solver.current_solution)
    print(f"Initial solution value: {initial_value}")
    print(f"initial solution is viable: {solver.is_solution_viable(solver.current_solution)}")

    solver.run(max_iter)

    print(f"After exchange, solution is viable: {solver.is_solution_viable(solver.current_solution)}")