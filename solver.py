from instance import Instance
from utils import distance
from copy import deepcopy
from collections import defaultdict


class Solver:
    def __init__(self, instance: Instance):
        self.instance = instance
        self.current_solution = self.generate_initial_solution()

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
    
    # Topological sort approach with Kahn's algorithm [https://en.wikipedia.org/wiki/Topological_sorting]
    def generate_initial_solution(self) -> list[int]:    
        prerequisites = deepcopy(self.instance.prerequisites)
        temples = self.instance.get_temples()

        # build reverse graph: t -> [who depends on t]
        dependents = defaultdict(list)
        for t, prereqs in prerequisites.items():
            for p in prereqs:
                dependents[p].append(t)

        # initial list of all temples without prerequisites
        no_prereqs = [t for t in temples if not prerequisites[t]]
        sol = []

        while no_prereqs:
            current = no_prereqs.pop()
            sol.append(current)

            for dep in dependents[current]:
                prerequisites[dep].remove(current)
                if not prerequisites[dep]:
                    no_prereqs.append(dep)

        if len(sol) != self.instance.num_temples:
            print(f"Initial solution does not contain all temples. ({len(sol)} / {self.instance.num_temples})")

        return sol

    
    ###########################
    #   Solution Evaluation   #
    ###########################
    
    """_summary_
    Args: solution (list[int]): The order in which temples are visited
    Returns: int: The total distance of the path
    
    Check if solution is valid BEFORE calling this function
    
    TODO: Implement memorization to avoid recalculating distances
    
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
        visited = dict.fromkeys(self.instance.get_temples(), False)

        if self.instance.get_prerequisites_for(solution[0]) != []:
            print(f"First temple {solution[0]} has prerequisites.")
            return False
        else:
            visited[solution[0]] = True

        for temple in solution[1:]:
            for prereq in self.instance.get_prerequisites_for(temple):
                if not visited[prereq]:
                    print(f"Temple {temple} visited before its prerequisite {prereq}.")
                    return False
            visited[temple] = True
            
        print('\n')
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


