import numpy as np
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict
from random import randint, sample, seed as set_random_seed
from copy import deepcopy
import multiprocessing
import time
import os

class Instance:
    num_temples: int
    num_prerequisites: int
    temples: dict[int, tuple[int, int]]
    prerequisites: dict[int, list[int]]
    def __init__(self, file_path: str):
        with open(file_path, "r") as input_file:
            lines = [line.strip() for line in input_file if line.strip()]
        self.num_temples = int(lines[0])
        self.num_prerequisites = int(lines[1 + self.num_temples])
        self.temples = {i + 1 : tuple(map(int, lines[i + 1].split())) for i in range(self.num_temples)}
        self.prerequisites = {t : [] for t in self.temples}
        for i in range(self.num_prerequisites):
            src, dst = map(int, lines[i + 2 + self.num_temples].split())
            self.prerequisites[dst].append(src)
    def get_temples(self) -> list[int]:
        return list(self.temples.keys())
    def get_temple(self, i:int) -> tuple[int, int]:
        return self.temples[i]
    def get_prerequisites_for(self, t:int) -> list[int]:
        return self.prerequisites.get(t)

# =============================================================================
# VNS SOLVER CLASS
# =============================================================================
class Solver:
    def __init__(self, instance: Instance):
        self.instance = instance
        self.temple_ids = np.array(sorted(self.instance.temples.keys()))
        self.temple_id_to_idx = {temple_id: i for i, temple_id in enumerate(self.temple_ids)}
        coords = np.array([self.instance.temples[tid] for tid in self.temple_ids])
        self.dist_matrix = squareform(pdist(coords, 'euclidean')).astype(float)
        initial_solution_ids = self.generate_initial_solution()
        self.current_solution = np.array([self.temple_id_to_idx[tid] for tid in initial_solution_ids], dtype=int)
        self.best_solution = self.current_solution.copy()
        self.current_solution_value = self.evaluate_solution(self.current_solution)
        self.best_solution_value = self.current_solution_value
        self.neighborhoods = [self.get_neighbors_1, self.get_neighbors_2, self.get_neighbors_3]
    
    def evaluate_solution(self, solution_indices: np.ndarray) -> float:
        return self.dist_matrix[solution_indices[:-1], solution_indices[1:]].sum()

    def shake(self, solution: np.ndarray, k: int) -> np.ndarray:
        shaken_sol = solution.copy()
        swaps_to_apply = self.exchange(shaken_sol, k)
        for i, j in swaps_to_apply:
            shaken_sol[[i, j]] = shaken_sol[[j, i]]
        return shaken_sol

    def get_neighbors_1(self, solution: np.ndarray):
        n = len(solution)
        for i in range(n):
            for j in range(i + 1, n):
                if self.swap_is_valid(solution, i, j):
                    yield (i, j)

    def get_neighbors_2(self, solution: np.ndarray):
        n = len(solution)
        yielded_swaps = set()
        for _ in range(n):
            try:
                i, j = sample(range(n), 2)
                swap_tuple = tuple(sorted((i, j)))
                if swap_tuple not in yielded_swaps and self.swap_is_valid(solution, swap_tuple[0], swap_tuple[1]):
                    yielded_swaps.add(swap_tuple)
                    yield swap_tuple
            except ValueError:
                continue

    def get_neighbors_3(self, solution: np.ndarray):
        n = len(solution)
        yielded_swaps = set()
        for _ in range(n * 2):
            try:
                i, j = sample(range(n), 2)
                swap_tuple = tuple(sorted((i, j)))
                if swap_tuple not in yielded_swaps and self.swap_is_valid(solution, swap_tuple[0], swap_tuple[1]):
                    yielded_swaps.add(swap_tuple)
                    yield swap_tuple
            except ValueError:
                continue
    
    def first_improvement(self, solution: np.ndarray, k_idx: int) -> np.ndarray:
        current_sol = solution.copy()
        current_value = self.evaluate_solution(current_sol)
        neighbors_generator = self.neighborhoods[k_idx]
        while True:
            improved = False
            for i, j in neighbors_generator(current_sol):
                current_sol[[i, j]] = current_sol[[j, i]]
                new_value = self.evaluate_solution(current_sol)
                if new_value < current_value:
                    current_value = new_value
                    improved = True
                    break
                else:
                    current_sol[[i, j]] = current_sol[[j, i]]
            if not improved:
                break
        return current_sol

    def swap_is_valid(self, solution_indices: np.ndarray, first_idx: int, second_idx: int) -> bool:
        id_at_first_pos = self.temple_ids[solution_indices[first_idx]]
        id_at_second_pos = self.temple_ids[solution_indices[second_idx]]
        if id_at_first_pos in self.instance.get_prerequisites_for(id_at_second_pos):
            return False
        if second_idx > first_idx + 1:
            nodes_between_ids = {self.temple_ids[solution_indices[k]] for k in range(first_idx + 1, second_idx)}
            prereqs_for_second_node = self.instance.get_prerequisites_for(id_at_second_pos)
            if not nodes_between_ids.isdisjoint(prereqs_for_second_node):
                return False
            for between_id in nodes_between_ids:
                if id_at_first_pos in self.instance.get_prerequisites_for(between_id):
                    return False
        return True
    
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

    def get_valid_swap(self, solution: np.ndarray) -> tuple[int, int]:
        n = len(solution)
        max_attempts = n * n
        for _ in range(max_attempts):
            try:
                i, j = sample(range(n), 2)
                first_idx, second_idx = min(i, j), max(i, j)
                if self.swap_is_valid(solution, first_idx, second_idx):
                    return first_idx, second_idx
            except ValueError:
                return None
        return None

    def exchange(self, solution: np.ndarray, k: int) -> list[tuple[int, int]]:
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
    
    def run(self, max_iter: int, verbose: bool = True):
        iter_count = 0
        max_k = len(self.neighborhoods)
        while iter_count < max_iter:
            k = 1
            while k <= max_k:
                shaken_solution = self.shake(self.current_solution, k)
                improved_solution = self.first_improvement(shaken_solution, k - 1)
                improved_value = self.evaluate_solution(improved_solution)
                if improved_value < self.current_solution_value:
                    self.current_solution = improved_solution
                    self.current_solution_value = improved_value
                    if verbose:
                        print(f"Iter {iter_count}: Found new solution with value {self.current_solution_value:.2f}. Resetting k=1.")
                    k = 1
                    if self.current_solution_value < self.best_solution_value:
                        self.best_solution = self.current_solution.copy()
                        self.best_solution_value = self.current_solution_value
                else:
                    k = k + 1
            iter_count += 1
        if verbose:
            print(f"\nFinished. Best solution value: {self.best_solution_value:.2f}")

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
            raise Exception("Could not generate a valid initial solution.")
        return sol

# =============================================================================
# WORKER FUNCTION FOR MULTIPROCESSING
# =============================================================================

def run_solver_instance(args):
    """
    A top-level function to be executed by each worker process.
    It initializes and runs one full VNS Solver.
    """
    file_path, max_iter, random_seed = args
    
    set_random_seed(random_seed)
    
    instance = Instance(file_path)
    solver = Solver(instance)
    
    solver.run(max_iter, verbose=False) 
    
    print(f"Process {os.getpid()} finished. Found best value: {solver.best_solution_value:.2f}")
    
    return (solver.best_solution_value, solver.best_solution, solver.temple_ids)