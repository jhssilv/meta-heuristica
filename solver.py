from instance import Instance
from utils import distance

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
    
    ################################
    #   First Solution Generation  #
    ################################
    
    def generate_initial_solution(self) -> list[int]:    
        initial_temple_idx = None
  
        # Search for a temple with no prerequisites to start with      
        for i in range(len(self.instance.temples)):
            if self.instances.prerequisites_map[i] == []:
                initial_temple_idx = i
                break        
        
        
              
            
    
    
    
    ###########################
    #   Solution Evaluation   #
    ###########################
    
    """_summary_
    Args: solution (list[int]): The order in which temples are visited
    Returns: int: The total distance of the path
    
    Check if solution is valid BEFORE calling this function
    
    """
    def evaluate_solution(self, solution: list[int]) -> int:
        return sum(
            distance(
                self.instance.get_temple(solution[i]), 
                self.instance.get_temple(solution[i+1])
            )
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


