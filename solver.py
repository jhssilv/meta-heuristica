from instance import Instance
import numpy as np

class Solver:
    def __init__(self, instance: Instance):
        self.instance = instance

    ###########################
    #   Local Search Methods  #
    ###########################

    def first_improvement(self):
        pass

    def best_improvement(self):
        pass
    
    ###########################
    #   Solution Evaluation   #
    ###########################
    
    """_summary_
    Args: solution (list[int]): The order in which temples are visited
    Returns: int: The total distance of the path
    
    Check if solution is valid BEFORE calling this function
    
    """
    def evaluate_solution(self, solution: list[int]) -> int:
        
        def distance(src: tuple[int, int], dst: tuple[int, int]) -> int:
            return np.floor(np.linalg.norm(np.array(src) - np.array(dst)) * 100)

        return sum(
            distance(
                self.instance.temples[solution[i]], 
                self.instance.temples[solution[i+1]]) 
            for i in range(len(solution) - 1)
            )

    ############################
    # Solution viability check #
    ############################

    """_summary_
    Args: solution (list[int]): The order in which temples are visited
    Returns: bool: Whether the solution is viable
    """
    def is_solution_viable(self, solution: list[int]) -> bool:
        visited = dict.fromkeys(range(self.instance.num_temples), False)

        if self.instance.prerequisites_map[solution[0]] != []:
            return False
        else:
            visited[solution[0]] = True

        for temple in solution[1:]:
            for prereq in self.instance.prerequisites_map[temple]:
                if not visited[prereq]:
                    return False
            visited[temple] = True
            
        return all(visited.values())

    ######################################
    # VNS (Variable Neighborhood Search) #
    ######################################

    def solve(self):
        pass


