from instance import Instance
from utils import distance
from copy import deepcopy
from collections import defaultdict
from random import randint
from time import time

class Solver:
    def __init__(self, instance: Instance):
        self.instance = instance
        self.current_solution = self.generate_initial_solution()
        self.current_solution_value = self.evaluate_solution(self.current_solution)

    ###########################
    #   Local Search Methods  #
    ###########################
    
    def swap(self, i:int, j:int):
        self.current_solution[i], self.current_solution[j] = self.current_solution[j], self.current_solution[i]
    
    def shake(self):
        swap_is_valid = False
        while not swap_is_valid:
            first_idx = randint(0, len(self.current_solution) - 1)
            second_idx = randint(0, len(self.current_solution) - 1)

            sol_first = self.current_solution[first_idx]
            sol_second = self.current_solution[second_idx]

            for i in range(first_idx, second_idx):
                swap_is_valid = True
                # if anything that depends on B comes after A or anything that B depends on comes after A, break
                if sol_first in self.instance.get_prerequisites_for(self.current_solution[i]) or \
                    self.current_solution[i] in self.instance.get_prerequisites_for(sol_second):
                        swap_is_valid = False
                        break
                    
        return (first_idx, second_idx)
                        
    def first_improvement(self, max_tries = 1000) -> bool:
        new_value = self.current_solution_value

        while(new_value >= self.current_solution_value):    
            delta = self.shake()
            new_value = self.evaluate_solution_delta(self.current_solution, delta)
            max_tries -= 1

        if max_tries > 0:
            self.swap(delta[0], delta[1])
            return True
        
        return False

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
        # this can be moved to instance if needed elsewhere
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
                self.instance.get_temple(solution[i + 1])
            )
            for i in range(len(solution) - 1)
            )

    def evaluate_solution_delta(self, solution: list[int], delta: tuple[int, int]) -> int:
        sol_value = self.current_solution_value
        
        # First we remove the distances from the edges that will be changed        
        
        if delta[0] < len(solution) - 1:
            sol_value -= distance(
                self.instance.get_temple(solution[delta[0]]),
                self.instance.get_temple(solution[delta[0] + 1])
            )
        
        if delta[1] < len(solution) - 1:    
            sol_value -= distance(
                self.instance.get_temple(solution[delta[1]]),
                self.instance.get_temple(solution[delta[1] + 1])
            )
        
        if delta[0] > 0:
            sol_value -= distance(
                self.instance.get_temple(solution[delta[0]]),
                self.instance.get_temple(solution[delta[0] - 1])
            )

        if delta[1] > 0:
            sol_value -= distance(
                self.instance.get_temple(solution[delta[1]]),
                self.instance.get_temple(solution[delta[1] - 1])
            )    
        
    # Then we add the distances from the new edges created by the swap
        if delta[0] < len(solution) - 1:    
            sol_value += distance(
                self.instance.get_temple(solution[delta[1]]),
                self.instance.get_temple(solution[delta[0] + 1])
            )
        if delta[1] < len(solution) - 1:    
            sol_value += distance(
                self.instance.get_temple(solution[delta[0]]),
                self.instance.get_temple(solution[delta[1] + 1])
            )

        if delta[0] > 0:
            sol_value += distance(
                self.instance.get_temple(solution[delta[1]]),
                self.instance.get_temple(solution[delta[0] - 1])
            )
            
        if delta[1] > 0:
            sol_value += distance(
                self.instance.get_temple(solution[delta[0]]),
                self.instance.get_temple(solution[delta[1] - 1])
            )
            
        return sol_value

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
                    return False
            visited[temple] = True
            
        if not all(visited.values()):
            return False            
        
        return True
    
    ######################################
    # VNS (Variable Neighborhood Search) #
    ######################################

    def solve(self):
        pass


