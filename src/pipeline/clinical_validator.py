import numpy as np
import csv
import os

class ClinicalValidator:
    def __init__(self):
        # Your specific Phase 5 architecture modalities
        self.modalities = [
            "chest_xray", 
            "brain_mri", 
            "skin_lesion", 
            "bone_fracture", 
            "ecg", 
            "retinal"
        ]
        print("[SYSTEM] Bare-Metal Clinical Validator Initialized (Pure Numpy).")

    def _compute_auc_roc(self, y_true, y_prob):
        """Custom pure-numpy AUC-ROC to avoid Sklearn dependency hell."""
        desc_indices = np.argsort(y_prob)[::-1]
        y_true_sorted = y_true[desc_indices]
        
        tps = np.cumsum(y_true_sorted)
        fps = np.cumsum(1 - y_true_sorted)
        
        if tps[-1] == 0 or fps[-1] == 0:
            return 0.5
            
        tpr = np.r_[0, tps / tps[-1]]
        fpr = np.r_[0, fps / fps[-1]]
        return np.trapz(tpr, fpr)

    def compute_metrics(self, y_true, y_pred, y_prob):
        """Computes critical medical metrics natively."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Calculate Confusion Matrix Elements
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        # Core Metrics (FDA/CE Standards)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
        auc = self._compute_auc_roc(y_true, y_prob)

        return {
            "Accuracy": round(accuracy, 4),
            "Sensitivity": round(sensitivity, 4),
            "Specificity": round(specificity, 4),
            "PPV": round(ppv, 4),
            "NPV": round(npv, 4),
            "AUC-ROC": round(auc, 4)
        }

    def generate_hospital_report(self, validation_data, output_dir="../../Final_Defense_Assets"):
        """Compiles metrics into a CSV using pure Python standard library."""
        print(f"\n[*] Generating Procurement Report for {len(self.modalities)} Modalities...")
        
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "clinical_validation_results.csv")
        
        headers = ["Modality", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "AUC-ROC"]
        
        print("="*85)
        print(f"{'MODALITY':<15} | {'ACC':<6} | {'SENS':<6} | {'SPEC':<6} | {'PPV':<6} | {'NPV':<6} | {'AUC-ROC':<6}")
        print("-" * 85)
        
        with open(report_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(headers)
            
            for modality in self.modalities:
                if modality not in validation_data:
                    continue
                    
                data = validation_data[modality]
                metrics = self.compute_metrics(data['y_true'], data['y_pred'], data['y_prob'])
                
                # Write to CSV
                writer.writerow([
                    modality.upper(), metrics["Accuracy"], metrics["Sensitivity"], 
                    metrics["Specificity"], metrics["PPV"], metrics["NPV"], metrics["AUC-ROC"]
                ])
                
                # Print to terminal
                print(f"{modality.upper():<15} | {metrics['Accuracy']:<6} | {metrics['Sensitivity']:<6} | {metrics['Specificity']:<6} | {metrics['PPV']:<6} | {metrics['NPV']:<6} | {metrics['AUC-ROC']:<6}")
                
        print("="*85)
        print(f"[SUCCESS] Official Hospital Report saved to: {report_path}")

if __name__ == "__main__":
    np.random.seed(42) 
    dummy_validation_data = {}
    validator = ClinicalValidator()
    
    for mod in validator.modalities:
        y_true = np.random.randint(0, 2, 1000)
        y_prob = np.where(y_true == 1, np.random.uniform(0.6, 1.0, 1000), np.random.uniform(0.0, 0.4, 1000))
        noise_indices = np.random.choice(1000, 50, replace=False)
        y_prob[noise_indices] = 1.0 - y_prob[noise_indices] 
        y_pred = (y_prob > 0.5).astype(int)
        
        dummy_validation_data[mod] = {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}
        
    validator.generate_hospital_report(dummy_validation_data)
