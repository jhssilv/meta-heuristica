
class Instance:
    num_temples: int                         # T
    num_prerequisites: int                   # P
    temples: list[tuple[int, int]]           # List of (Xt, Yt) coordinates of temples (this is only used for distance calculations)
    prerequisites: list[tuple[int, int]]     # List of (A, B) pairs where A is prerequisite for B

    prerequisites_map = dict[int, list[int]] # Map of temple index to list of prerequisite temple indices
                                             # Use this to look up prerequisites more efficiently
                                             # Each value here refers to the index of the temple in the `temples` list

    def __init__(self, file_path: str):
        """Load instance data from a file."""
        print(f"Loading instance from {file_path}...")

        with open(file_path, "r") as input_file:
            lines = [line.strip() for line in input_file if line.strip()]

        self.num_temples = int(lines[0])
        self.num_prerequisites = int(lines[1 + self.num_temples])

        print(f"Loading {self.num_temples} temples...")
        self.temples = [tuple(map(int, lines[i + 1].split())) # split the line into two integers
                        for i in range(self.num_temples)]     # and use map to convert them to int

        print(f"Loading {self.num_prerequisites} prerequisites...")
        self.prerequisites = [ tuple(map(int, lines[i + 2 + self.num_temples].split())) 
                              for i in range(self.num_prerequisites)]

        print("Mapping prerequisites...")
        self.prerequisites_map = dict.fromkeys(range(self.num_temples), [])
        for src, dst in self.prerequisites:
            self.prerequisites_map[dst].append(src)

        print("Instance loaded successfully.")
