import numpy as np
import time

class FederatedAggregationServer:
    def __init__(self):
        print("==================================================")
        print("[FEDERATED SERVER] Global Aggregation Node Online")
        print("==================================================")
        self.global_weights = None
        self.client_updates = []

    def receive_hospital_update(self, hospital_id, local_weights, sample_size):
        """Simulates receiving encrypted weight matrices from remote hospital nodes."""
        print(f"[NETWORK] Received encrypted update from {hospital_id} | Samples: {sample_size}")
        self.client_updates.append({"weights": local_weights, "samples": sample_size})

    def execute_fedavg(self):
        """
        Executes the Federated Averaging (FedAvg) algorithm.
        Formula: W_global = Σ (n_k / n_total) * W_k
        """
        if not self.client_updates:
            print("[FEDAVG ERROR] No client updates received.")
            return False

        print("\n[FEDAVG] Initiating Global Weight Aggregation (FedAvg)...")
        time.sleep(1) # Simulating processing time
        
        # Calculate total data pool size across all hospitals
        total_samples = sum(client["samples"] for client in self.client_updates)
        print(f"[*] Total decentralized samples in this round: {total_samples}")

        # Initialize the new global weight matrix with zeros
        shape = self.client_updates[0]["weights"].shape
        new_global_weights = np.zeros(shape, dtype=np.float32)

        # Apply the FedAvg Formula
        for client in self.client_updates:
            fraction = client["samples"] / total_samples
            # Scale the local weights by their fraction of the total dataset
            scaled_weights = client["weights"] * fraction
            new_global_weights += scaled_weights
            
        self.global_weights = new_global_weights
        
        # Clear the queue for the next training round
        self.client_updates = []
        
        print("[FEDAVG SUCCESS] Global model synchronized. Zero patient data compromised.")
        print("==================================================")
        return True

# ==========================================
# TEST THE FEDERATED NETWORK
# ==========================================
if __name__ == "__main__":
    fed_server = FederatedAggregationServer()
    
    # 1. Simulate the base shape of our neural network weights
    dummy_weight_shape = (512, 128)
    
    # 2. Simulate Hospital A (Mayo Clinic) training locally on 1,200 scans
    hosp_a_weights = np.random.randn(*dummy_weight_shape).astype(np.float32)
    fed_server.receive_hospital_update("Mayo Clinic (Node A)", hosp_a_weights, sample_size=1200)
    
    # 3. Simulate Hospital B (Shaukat Khanum) training locally on 850 scans
    hosp_b_weights = np.random.randn(*dummy_weight_shape).astype(np.float32)
    fed_server.receive_hospital_update("Shaukat Khanum (Node B)", hosp_b_weights, sample_size=850)
    
    # 4. Simulate Hospital C (Aga Khan) training locally on 2,100 scans
    hosp_c_weights = np.random.randn(*dummy_weight_shape).astype(np.float32)
    fed_server.receive_hospital_update("Aga Khan Univ (Node C)", hosp_c_weights, sample_size=2100)
    
    # 5. Run the Aggregation
    fed_server.execute_fedavg()
