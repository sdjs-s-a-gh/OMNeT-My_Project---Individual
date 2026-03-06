class RLResourceAllocator():
    def __init__(self):
        print("Python: RL Agent has been initialised")
        
    def decide_allocation(self, current_load):
        print(f"Python Received load {current_load}.")
        return 1 if current_load > 0.5 else 0
    
    def allocate_resources(self, required_cpu_cycles):
        return required_cpu_cycles