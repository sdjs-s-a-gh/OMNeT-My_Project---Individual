class RLResourceAllocator():
    def __init__(self):
        print("Python: RL Agent has been initialised")
        
    def decide_allocation(self, current_load):
        print(f"Python Received load {current_load}.")
        return 1 if current_load > 0.5 else 0
    
    def allocate_resources_static(self, required_cpu_cycles):
        """
        A dummy function used just to ensure that this script can be accessed from the Simulator.
        """
        return required_cpu_cycles
    
    def allocate_resources_ppo_dummy(self, max_cpu_capacity, required_cpu_cycles, resource_utilisation):
        """
        Input:
            # CPU Cycles Required
            # Maybe deadline latency
            
        State: 
            # Network Conditions (connection quality expressed as SINR (Alex); Path loss (Chen); Packet loss, communication latency, bandwidth (Mahimalmur))
            # Queue Length (Liu, Mahimalmur)
            # Waiting time for pending tasks (Mahimalmur)
            # Resource Utilisation (Mahimalur) or Availability (Liu)
            # Current Latency of the system (which would just be the average of each task combined)
            # Current Energy Consumption of the system (which would just be the average of each task combined) (Alex, Mahimalmur)            
        
        Action/Output:
            # Allocate x number of CPU cycles to the task.
        """
        print(f"Max CPU Capacity: {max_cpu_capacity}; Required CPU Cycles: {required_cpu_cycles}; Resource Utilisation: {resource_utilisation}")
        if resource_utilisation < 0.5:
            return int(max_cpu_capacity/2)
        else:
            return required_cpu_cycles
        
    def allocate_resources(self, required_cpu_cycles, communication_latency, resource_utilisation, queue_length, total_cpu_cycles_in_queue):
        """
        args: 
        * communication_latency: Measured in milliseconds, the time delay between the task leaving the sending device and arriving on the edge server.
        """
        print(f"Required CPU Cycles: {required_cpu_cycles}; Communication Latency: {communication_latency}; Resource Utilisation: {resource_utilisation}; Queue Length: {queue_length}; CPU Cycles in Queue: {total_cpu_cycles_in_queue}")
        return required_cpu_cycles
        
        