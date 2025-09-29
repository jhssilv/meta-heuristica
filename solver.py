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
    
    # Topological sort approach [https://en.wikipedia.org/wiki/Topological_sorting]
    def generate_initial_solution(self) -> list[int]:    
        no_prereqs_idx = [] 
        sol = []
        
        initial_temple_idx = None
  
        # Search for a temple with no prerequisites to start with      
        for i in range(1, len(self.instance.temples) + 1):
            if self.instance.get_prerequisites_for(i) == []:
                no_prereqs_idx.append(i)
        
        while no_prereqs_idx:
            current = no_prereqs_idx.pop()
            sol.append(current)
            
            for neighbor in range(1, len(self.instance.temples) + 1):
                if current in self.instance.get_prerequisites_for(neighbor):  # This can be optimized if we create
                    self.instance.prerequisites_map[neighbor].remove(current) # a map from temple to its dependents
                    if self.instance.get_prerequisites_for(neighbor) == []:
                        no_prereqs_idx.append(neighbor)
            
            
        return sol
    
    
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
        visited = dict.fromkeys(range(1,self.instance.num_temples + 1), False)

        if self.instance.prerequisites_map[solution[0]] != []:
            return False
        else:
            visited[solution[0]] = True

        for temple in solution[1:]:
            for prereq in self.instance.prerequisites_map[temple]:
                if not visited[prereq]:
                    print(f"Temple {temple} visited before its prerequisite {prereq}.")
                    return False
            visited[temple] = True
            
        if not all(visited.values()):
            for temple, was_visited in visited.items():
                if not was_visited:
                    print(f"Temple {temple} was never visited.")
            return False            
        
        return True

    ######################################
    # VNS (Variable Neighborhood Search) #
    ######################################

    def solve(self):
        pass


