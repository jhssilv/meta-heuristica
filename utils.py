import numpy as np

def distance(src: tuple[int, int], dst: tuple[int, int]) -> int:
    return np.floor(np.linalg.norm(np.array(src) - np.array(dst)) * 100)

def swap(solution: list[int], i: int, j: int) -> None:
    solution[i], solution[j] = solution[j], solution[i]