import numpy as np
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict
from random import randint, sample, seed as random_seed
from copy import deepcopy

from instance import Instance

class Solver:
    def __init__(self, instance: Instance, seed: int = None, show_simplified_output: bool = False):
        self.instance = instance
        self.seed = seed
        self.show_simplified_output = show_simplified_output

        if seed is not None:
            random_seed(seed)
            np.random.seed(seed)

        self.temple_ids = np.array(sorted(self.instance.temples.keys()))
        self.temple_id_to_idx = {temple_id: i for i, temple_id in enumerate(self.temple_ids)}
        
        coords = np.array([self.instance.temples[tid] for tid in self.temple_ids])

        self.dist_matrix = np.floor(squareform(pdist(coords, 'euclidean')) * 100)

        initial_solution_ids = self.generate_initial_solution()
        self.current_solution = np.array([self.temple_id_to_idx[tid] for tid in initial_solution_ids], dtype=int)
        
        self.best_solution = self.current_solution.copy()
        
        self.current_solution_value = self.evaluate_solution(self.current_solution)
        self.best_solution_value = self.current_solution_value
        
        self.neighborhoods = [self.get_neighbors_1, self.get_neighbors_2, self.get_neighbors_3]
    
    ############################
    #   Solution Validation    #
    ############################
    
    def is_solution_viable(self, solution_indices: np.ndarray) -> bool:
        """
        Checks if a solution is viable by ensuring all prerequisites are met in order.
        """
        visited_ids = set()
        for temple_idx in solution_indices:
            temple_id = self.temple_ids[temple_idx]
            
            prerequisites = self.instance.get_prerequisites_for(temple_id)
            if not set(prerequisites).issubset(visited_ids):
                print(f"Error: Temple {temple_id} is visited before its prerequisite.")
                return False
            
            visited_ids.add(temple_id)
            
        return True

    ###########################
    #   Solution Evaluation   #
    ###########################

    def evaluate_solution(self, solution_indices: np.ndarray) -> float:
        """
        Calculates the total path distance using the pre-computed matrix.
        This is fully vectorized and extremely fast.
        """
        return self.dist_matrix[solution_indices[:-1], solution_indices[1:]].sum()

    ###########################
    #       Shake             #
    ###########################

    def shake(self, solution: np.ndarray, k: int) -> np.ndarray:
        """
        Shakes a solution by applying k random valid swaps.
        Returns a new, shaken solution array.
        """
        shaken_sol = solution.copy()
        swaps_to_apply = self.exchange(shaken_sol, k)
        for i, j in swaps_to_apply:
            shaken_sol[[i, j]] = shaken_sol[[j, i]]
        return shaken_sol

    ###############################
    #   Neighborhood Generators   #
    ###############################

    def get_neighbors_1(self, solution: np.ndarray):
        """
        NEIGHBORHOOD 1: Systematic 2-opt.
        Generator that yields all valid swaps by iterating through all pairs.
        This is the most thorough but slowest local search neighborhood.
        """
        n = len(solution)
        for i in range(n):
            for j in range(i + 1, n):
                if self.swap_is_valid(solution, i, j):
                    yield (i, j)

    def get_neighbors_2(self, solution: np.ndarray):
        """
        NEIGHBORHOOD 2: Randomized 2-opt.
        Generator that yields up to N random valid swaps.
        Faster than systematic search, good for diversification.
        """
        n = len(solution)
        yielded_swaps = set()
        for _ in range(n):
            try:
                i, j = sample(range(n), 2)
                first_idx, second_idx = min(i, j), max(i, j)
                swap_tuple = (first_idx, second_idx)
                
                if swap_tuple not in yielded_swaps and self.swap_is_valid(solution, first_idx, second_idx):
                    yielded_swaps.add(swap_tuple)
                    yield swap_tuple
            except ValueError:
                continue

    def get_neighbors_3(self, solution: np.ndarray):
        """
        NEIGHBORHOOD 3: Aggressive Randomized 2-opt.
        Generator that yields up to N*2 random valid swaps.
        Provides a larger jump to escape tricky local optima.
        """
        n = len(solution)
        yielded_swaps = set()
        for _ in range(n * 2):
            try:
                i, j = sample(range(n), 2)
                first_idx, second_idx = min(i, j), max(i, j)
                swap_tuple = (first_idx, second_idx)
                
                if swap_tuple not in yielded_swaps and self.swap_is_valid(solution, first_idx, second_idx):
                    yielded_swaps.add(swap_tuple)
                    yield swap_tuple
            except ValueError:
                continue

    ###########################
    #   Local Search Methods  #
    ###########################

    def first_improvement(self, solution: np.ndarray, k: int) -> np.ndarray:
        """
        Applies local search until a local optimum is found.
        Returns a new, improved solution array.
        """
        current_sol = solution.copy()
        current_value = self.evaluate_solution(current_sol)
        
        n = len(current_sol)
        neighbors_generator = self.neighborhoods[k - 1]

        while True:
            improved = False
            best_delta = 0
            best_swap = None

            for i, j in neighbors_generator(current_sol):
                if i > j: i, j = j, i

                node_i, node_j = current_sol[i], current_sol[j]
                delta = 0

                if j == i + 1:
                # --- CASO 1: Nós adjacentes ---
                # Tratando casos de borda (início e fim da rota)
                    if i == 0: # Troca no início
                        cost_before = self.dist_matrix[node_j, current_sol[j+1]]
                        cost_after = self.dist_matrix[node_i, current_sol[j+1]]
                    elif j == n - 1: # Troca no fim
                        cost_before = self.dist_matrix[current_sol[i-1], node_i]
                        cost_after = self.dist_matrix[current_sol[i-1], node_j]
                    else: # Troca no meio
                        cost_before = self.dist_matrix[current_sol[i-1], node_i] + self.dist_matrix[node_j, current_sol[j+1]] if i > 0 and j < n - 1 else 0
                        cost_after = self.dist_matrix[current_sol[i-1], node_j] + self.dist_matrix[node_i, current_sol[j+1]] if i > 0 and j < n - 1 else 0

                    delta = cost_after - cost_before
                else:
                    # --- CASO 2: Nós não-adjacentes ---

                    # Tratando casos de borda
                    if i == 0:
                        cost_removed = self.dist_matrix[node_i, current_sol[i+1]] + self.dist_matrix[current_sol[j-1], node_j] + self.dist_matrix[node_j, current_sol[j+1]]
                        cost_added = self.dist_matrix[node_j, current_sol[i+1]] + self.dist_matrix[current_sol[j-1], node_i] + self.dist_matrix[node_i, current_sol[j+1]]
                    elif j == n - 1:
                        cost_removed = self.dist_matrix[current_sol[i-1], node_i] + self.dist_matrix[node_i, current_sol[i+1]] + self.dist_matrix[current_sol[j-1], node_j]
                        cost_added = self.dist_matrix[current_sol[i-1], node_j] + self.dist_matrix[node_j, current_sol[i+1]] + self.dist_matrix[current_sol[j-1], node_i]
                    else:
                        cost_removed = self.dist_matrix[current_sol[i-1], node_i] + self.dist_matrix[node_i, current_sol[i+1]] + \
                                    self.dist_matrix[current_sol[j-1], node_j] + self.dist_matrix[node_j, current_sol[j+1]]
                        
                        cost_added =  self.dist_matrix[current_sol[i-1], node_j] + self.dist_matrix[node_j, current_sol[i+1]] + \
                                    self.dist_matrix[current_sol[j-1], node_i] + self.dist_matrix[node_i, current_sol[j+1]]
                    
                    delta = cost_added - cost_removed

                if delta < 0:
                    # Encontrou a primeira melhora, aplica a troca e reinicia a busca
                    current_sol[[i, j]] = current_sol[[j, i]]
                    current_value += delta
                    improved = True
                    break
            
            if not improved:
                break
        
        return current_sol

    ###########################
    #      K-Exchange &       #
    #   Prerequisite Checks   #
    ###########################

    def swap_is_valid(self, solution_indices: np.ndarray, first_idx: int, second_idx: int) -> bool:
        """
        Checks if swapping two nodes at the given indices is valid by respecting ALL prerequisite constraints.
        This is a corrected and more robust version.
        """
        if first_idx > second_idx:
            first_idx, second_idx = second_idx, first_idx

        id_at_first_pos = self.temple_ids[solution_indices[first_idx]]
        id_at_second_pos = self.temple_ids[solution_indices[second_idx]]

        # --- RULE 1: Direct dependency check ---
        if id_at_first_pos in self.instance.get_prerequisites_for(id_at_second_pos):
            return False

        # --- RULE 2: Check all nodes BETWEEN the swapped pair ---
        if second_idx > first_idx + 1:
            nodes_between_ids = {self.temple_ids[solution_indices[k]] for k in range(first_idx + 1, second_idx)}

            prereqs_for_second_node = self.instance.get_prerequisites_for(id_at_second_pos)
            if not nodes_between_ids.isdisjoint(prereqs_for_second_node):
                return False

            for between_id in nodes_between_ids:
                if id_at_first_pos in self.instance.get_prerequisites_for(between_id):
                    return False
        return True

    def get_valid_swap(self, solution: np.ndarray) -> tuple[int, int]:
        """Finds a single random, valid swap."""
        n = len(solution)
        max_attempts = n * n 
        for _ in range(max_attempts):
            i, j = sample(range(n), 2)
            first_idx, second_idx = min(i, j), max(i, j)
            if self.swap_is_valid(solution, first_idx, second_idx):
                return first_idx, second_idx
        return None 

    def exchange(self, solution: np.ndarray, k: int) -> list[tuple[int, int]]:
        """Generates k valid swaps without applying them."""
        swaps = []
        temp_sol = solution.copy()
        count = k
        while count > 0:
            swap_pair = self.get_valid_swap(temp_sol)
            if swap_pair is None:
                break
            first_idx, second_idx = swap_pair
            temp_sol[[first_idx, second_idx]] = temp_sol[[second_idx, first_idx]]
            swaps.append((first_idx, second_idx))
            count -= 1
        return swaps

    ######################################
    # VNS (Variable Neighborhood Search) #
    ######################################
    
    def run(self, max_iter: int) -> None:
        iter_count = 0
        max_k = len(self.neighborhoods)

        # Show initial solution
        self.print_solution_details(self.current_solution, "Initial Solution")

        while iter_count < max_iter:
            k = 1
            while k <= max_k:
                # Shake
                shaken_solution = self.shake(self.current_solution, k)
                
                # Local Search
                improved_solution = self.first_improvement(shaken_solution, k - 1)
                improved_value = self.evaluate_solution(improved_solution)
                
                # Move 
                if improved_value < self.current_solution_value:
                    self.current_solution = improved_solution
                    self.current_solution_value = improved_value
                    if self.show_simplified_output:
                        print(f"Iter {iter_count}: Found new solution with value {self.current_solution_value:.2f}.")
                    else:
                        self.print_solution_details(self.current_solution, "Current Solution")
                    k = 1
                    
                    if self.current_solution_value < self.best_solution_value:
                        self.best_solution = self.current_solution.copy()
                        self.best_solution_value = self.current_solution_value
                else:
                    k = k + 1

            iter_count += 1
        
        final_solution_ids = [self.temple_ids[i] for i in self.best_solution]
        print(f"\nFinished. Best solution value: {self.best_solution_value:.2f}")
        self.print_solution_details(self.best_solution, "Final Best Solution")
    
    ################################
    #   First Solution Generation  #
    ################################
            
    def generate_initial_solution(self) -> list[int]:    
        prerequisites = deepcopy(self.instance.prerequisites)
        temples = self.instance.get_temples()
        dependents = defaultdict(list)
        for t, prereqs in prerequisites.items():
            for p in prereqs:
                dependents[p].append(t)
        no_prereqs = [t for t in temples if not prerequisites[t]]
        sol = []
        while no_prereqs:
            current = no_prereqs.pop(randint(0, len(no_prereqs) - 1))
            sol.append(current)
            for dep in dependents.get(current, []):
                prerequisites[dep].remove(current)
                if not prerequisites[dep]:
                    no_prereqs.append(dep)
        if len(sol) != self.instance.num_temples:
            raise Exception("Could not generate a valid initial solution. Check for cycles in prerequisites.")
        return sol
    
    def print_solution_details(self, solution_indices: np.ndarray, title: str = "Solution"):
        """
        Prints detailed information about a solution in a readable format.
        """
        solution_ids = [self.temple_ids[i] for i in solution_indices]
        solution_value = self.evaluate_solution(solution_indices)
        is_valid = self.is_solution_viable(solution_indices)
        
        print(f"\n{'='*50}")
        print(f"{title.upper()}")
        print(f"{'='*50}")
        print(f"Value: {solution_value:.2f}")
        print(f"Valid: {is_valid}")
        print(f"Number of temples: {len(solution_ids)}")
        
        print(f"\nVisit Order:")
        print("-" * 30)
        for i, temple_id in enumerate(solution_ids, 1):
            coords = self.instance.temples[temple_id]
            prereqs = self.instance.get_prerequisites_for(temple_id)
            prereq_str = f" (after: {prereqs})" if prereqs else " (no prereqs)"
            print(f"{i:3d}. Temple {temple_id:3d} at {coords}{prereq_str}")
        
        # Calculate distances between consecutive temples
        print(f"\nPath Distances:")
        print("-" * 30)
        total_distance = 0
        for i in range(len(solution_indices) - 1):
            from_idx, to_idx = solution_indices[i], solution_indices[i + 1]
            distance = self.dist_matrix[from_idx, to_idx]
            from_id, to_id = self.temple_ids[from_idx], self.temple_ids[to_idx]
            print(f"Temple {from_id:3d} → Temple {to_id:3d}: {distance:6.0f}")
            total_distance += distance
        
        print(f"\nTotal Distance: {total_distance:.0f}")
        print(f"{'='*50}")