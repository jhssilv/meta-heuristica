
class Instance:
    num_temples: int                         # T
    num_prerequisites: int                   # P

    temples: dict[int, tuple[int, int]]  # temple_id -> (x, y)
    prerequisites: dict[int, list[int]]  # temple_id -> [prerequisite_temple_ids]

    def __init__(self, file_path: str):
        """Load instance data from a file."""
        print(f"Loading instance from {file_path}...")

        with open(file_path, "r") as input_file:
            lines = [line.strip() for line in input_file if line.strip()]

        self.num_temples = int(lines[0])
        self.num_prerequisites = int(lines[1 + self.num_temples])

        print(f"Loading {self.num_temples} temples...")
        self.temples = {i + 1 : tuple(map(int, lines[i + 1].split())) # split the line into two 'integers'
                        for i in range(self.num_temples)}         # and use map to convert them to int
        
        #for entry in self.temples.items():
        #    print(entry)

        print(f"Loading {self.num_prerequisites} prerequisites...")
        self.prerequisites = {t : [] for t in self.temples}
        
        for i in range(self.num_prerequisites):
            src, dst = map(int, lines[i + 2 + self.num_temples].split())
            self.prerequisites[dst].append(src)

        print("Instance loaded successfully.")
        
    def get_temples(self) -> list[int]:
        return list(self.temples.keys())
        
    def get_temple(self, i:int) -> tuple[int, int]:
        return self.temples[i] # returns (x, y)

    def get_prerequisites_for(self, t:int) -> list[int]:
        return self.prerequisites.get(t)
