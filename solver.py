from instance import Instance
from utils import distance, swap
from copy import deepcopy
from collections import defaultdict
from random import randint, sample

class Solver:
    def __init__(self, instance: Instance):
        self.instance = instance
        
        self.current_solution = self.generate_initial_solution()
        self.best_solution = [i for i in self.current_solution]
        
        self.current_solution_value = self.evaluate_solution(self.current_solution)
        self.best_solution_value = self.current_solution_value
        
        self.neighborhoods = [self.get_neighbors_1, self.get_neighbors_2, self.get_neighbors_3]
        

    ###########################
    #       Shake             #
    ###########################

    def shake(self, solution: list[int], k: int) -> None:
        delta = self.exchange(solution, k)
        return delta

    ###############################
    #   Neighborhood Generators   #
    ###############################

    # neighborhoods shold be ordered by increasing complexity / size
    # neighborhood 1 is the smallest, neighborhood 3 is the largest
    # the more complex the neighborhood, the farther the solution will jump

    # TODO: implement get_neighbors_2 and get_neighbors_1 (they are currently copies of get_neighbors_3)

    # Generates n² random swaps 
    def get_neighbors_3(self, solution: list[int]) -> tuple[int, int]:
        indices = sample(list(range(len(solution))), len(solution))
        neighborhood = []

        for i in indices:
            for j in indices:
                first_index = min(i,j)
                second_index = max(i,j)
                if self.swap_is_valid(solution, first_index, second_index):
                    swap(solution, first_index, second_index)
                    neighborhood.append((first_index, second_index))

        for n in neighborhood[::-1]:
            swap(solution, n[0], n[1])

        return neighborhood

    def get_neighbors_2(self, solution: list[int]) -> list[tuple[int, int]]:
        indices = sample(list(range(len(solution))), len(solution))
        neighborhood = []

        for i in indices:
            for j in indices:
                first_index = min(i,j)
                second_index = max(i,j)
                if self.swap_is_valid(solution, first_index, second_index):
                    swap(solution, first_index, second_index)
                    neighborhood.append((first_index, second_index))

        for n in neighborhood[::-1]:
            swap(solution, n[0], n[1])

        return neighborhood
    
    def get_neighbors_1(self, solution: list[int]) -> list[tuple[int, int]]:
        indices = sample(list(range(len(solution))), len(solution))
        neighborhood = []

        for i in indices:
            for j in indices:
                first_index = min(i,j)
                second_index = max(i,j)
                if self.swap_is_valid(solution, first_index, second_index):
                    swap(solution, first_index, second_index)
                    neighborhood.append((first_index, second_index))

        for n in neighborhood[::-1]:
            swap(solution, n[0], n[1])

        return neighborhood

    ###########################
    #   Local Search Methods  #
    ###########################

    def first_improvement(self, sol: list[int], delta: list[tuple[int, int]], k:int) -> list[tuple[int, int]]:
        swaps = []
        
        # value of the solution A the shake
        shaken_value = self.evaluate_solution_delta(sol, delta)
        
        # apply the shake
        for d in delta:
            swap(sol, d[0], d[1])
        
        neighbors_generator = self.neighborhoods[k - 1]
        
        improved = False
        while not improved:
            improved = True
            
            for n in neighbors_generator(sol):
                new_value = self.evaluate_solution_delta(sol, [n])
                swap(sol, n[0], n[1])
                swaps.append(n)
                if new_value < shaken_value:
                    improved = True
                    break
                
        for s in (delta + swaps)[::-1]:
            swap(sol, s[0], s[1])
                
        return swaps

    ###########################
    #         K-Exchange      #
    ###########################

    def swap_is_valid(self, solution:list[int], first_idx:int, second_idx:int) -> bool:
        if first_idx < 0 or second_idx < 0: 
            return False

        for i in range(first_idx, second_idx):
            if solution[first_idx] in self.instance.get_prerequisites_for(solution[i]) or \
                solution[i] in self.instance.get_prerequisites_for(solution[second_idx]):
                    return False
        return True

    def get_valid_swap(self, solution:list[int]) -> tuple[int,int] | None:
        valid_swap = False
        while(not valid_swap):
            i = randint(0, len(solution) - 1)
            j = randint(0, len(solution) - 1)

            if i == j: 
                continue

            first_idx = min(i,j)
            second_idx = max(i,j)

            valid_swap = self.swap_is_valid(solution, first_idx, second_idx)

        return (first_idx, second_idx)

    # Create k valid swaps and don't apply them
    def exchange(self,solution:list[int], k:int) -> list[tuple[int, int]]:
        swaps = []
        while  k > 0:
            first_idx, second_idx = self.get_valid_swap(solution)
            swap(solution, first_idx, second_idx)
            swaps.append((first_idx, second_idx))
            k -= 1

        # removes the swaps
        for s in swaps[::-1]:
            swap(solution, s[0], s[1])

        return swaps

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

    def evaluate_solution_delta(self, solution: list[int], deltas: list[tuple[int, int]]) -> int:
        sol_value = self.current_solution_value
        
        # First we remove the distances from the edges that will be changed        
        
        for delta in deltas:
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
        for delta in deltas:
    
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

    def run(self, max_iter: int) -> None:
        iter = 0
        max_k = 3
        while iter < max_iter:
            k = 1
            while k <= max_k:
                print(self.is_solution_viable(self.current_solution), self.best_solution_value)
                delta = self.shake(self.best_solution, k)                         # s' <- shake(s, k)
                delta2 = self.first_improvement(self.current_solution, delta, k)  # s'' <- first_improvement(s', N_k)
                
                new_value = self.evaluate_solution_delta(self.current_solution, delta + delta2)
                
                if self.current_solution_value < new_value:        # s' worse than s (s < s')
                    k = k + 1                                      # k <- k + 1
                else:
                    self.current_solution_value = new_value        # s <- s'
                    for d in delta + delta2:
                        swap(self.current_solution, d[0], d[1])    # k <- 1
                    k = 1
                if new_value < self.best_solution_value:           
                    self.best_solution_value = new_value                     
                    self.best_solution = [i for i in self.current_solution] # s* <- s
            iter += 1