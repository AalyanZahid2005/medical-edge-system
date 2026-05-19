import json
import requests
import datetime
try:
    # This works when running the main pipeline
    from pipeline.icd10_mapper import ICD10Mapper
except ModuleNotFoundError:
    # This works when testing the script directly in the terminal
    from icd10_mapper import ICD10Mapper

class FHIRBridge:
    def __init__(self, server_url="http://192.168.1.100:8080/fhir"):
        """
        Initializes the FHIR Bridge. 
        Change the server_url to the IP address of your Orthanc/HAPI laptop.
        """
        self.icd10 = ICD10Mapper()
        self.server_url = server_url.rstrip('/')
        print(f"[FHIR BRIDGE] Initialized. Target EMR Server: {self.server_url}")

    def _get_loinc_code(self, modality):
        """Maps your hardware modalities to standard LOINC radiology codes."""
        codes = {
            "mri": {"code": "24534-0", "display": "MRI Head"},
            "chest": {"code": "36643-5", "display": "Chest X-ray"},
            "retinal": {"code": "70921-2", "display": "US Eye"},
            "ecg": {"code": "34534-8", "display": "EKG 12 channel"},
            "bone": {"code": "36642-7", "display": "Bone X-ray"},
            "skin": {"code": "59284-0", "display": "Dermatology morphology"}
        }
        return codes.get(modality.lower(), {"code": "18748-4", "display": "Diagnostic imaging study"})

    def create_diagnostic_report(self, patient_id, modality, diagnosis, confidence, llm_report):
        """
        Translates raw AI pipeline data into a valid HL7 FHIR R4 DiagnosticReport JSON.
        """
        loinc = self._get_loinc_code(modality)
        
        # FHIR R4 ISO-8601 Timestamp
        now_iso = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        # Get the billing code
        icd10_data = self.icd10.get_code(diagnosis, confidence)
        # Constructing the exact FHIR R4 Resource JSON
        fhir_resource = {
            "resourceType": "DiagnosticReport",
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                            "code": "RAD",
                            "display": "Radiology"
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": loinc["code"],
                        "display": loinc["display"]
                    }
                ]
            },
            "subject": {
                "display": f"Anoymous Edge Patient {patient_id}" 
            },
            "effectiveDateTime": now_iso,
            "conclusion": llm_report, # The localized text from your Phi-2/Template engine
            "conclusionCode": [
                {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "display": f"AI Finding: {diagnosis} (Confidence: {confidence:.1f}%)"
                        },
                        {
                            "system": "http://hl7.org/fhir/sid/icd-10-cm",
                            "code": icd10_data["code"],
                            "display": f"ICD-10: {icd10_data['description']}"
                        }
                    ]
                }
            ]
        }        
        return fhir_resource

    def transmit_to_emr(self, fhir_resource):
        """
        POSTs the FHIR payload to the HAPI FHIR server.
        """
        headers = {'Content-Type': 'application/fhir+json'}
        endpoint = f"{self.server_url}/DiagnosticReport"
        
        print("\n[FHIR BRIDGE] Transmitting DiagnosticReport to EMR...")
        
        try:
            response = requests.post(endpoint, headers=headers, data=json.dumps(fhir_resource), timeout=5)
            
            if response.status_code in [200, 201]:
                print(f"[SUCCESS] HL7 FHIR Resource accepted by server.")
                print(f"[NETWORK] Server Response Code: {response.status_code}")
                # HAPI FHIR returns the created ID in the Location header or body
                created_id = response.json().get('id', 'Unknown')
                print(f"[SYSTEM] Assigned EMR Document ID: DiagnosticReport/{created_id}")
                return True
            else:
                print(f"[ERROR] FHIR Server rejected payload. Code: {response.status_code}")
                print(response.text)
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"[CRITICAL] Connection refused. Is the HAPI FHIR server running on {self.server_url}?")
            return False
        except requests.exceptions.Timeout:
            print("[CRITICAL] Transmission timed out.")
            return False

# ==========================================
# TEST BED
# ==========================================
if __name__ == "__main__":
    print("============================================================")
    print("[SYSTEM] Testing Edge-to-EMR FHIR Integration...")
    print("============================================================\n")

    # 1. Initialize bridge (Point this to your Orthanc laptop's IP)
    # Example: If your laptop is 192.168.1.45, put "http://192.168.1.45:8080/fhir"
    bridge = FHIRBridge(server_url="http://hapi.fhir.org/baseR4") 
    
    # 2. Mock data from your pipeline
    sample_modality = "mri"
    sample_diagnosis = "Glioma"
    sample_confidence = 98.0
    sample_llm_text = "The patient presents with a high-risk diagnosis of a malignant brain tumor. The AI has a high level of confidence in its analysis and recommends immediate surgical intervention."
    
    # We use Patient/1 as a dummy for testing. In a real hospital, this is fetched via barcode/RFID.
    test_patient_id = "1" 

    # 3. Compile the FHIR JSON
    fhir_payload = bridge.create_diagnostic_report(
        patient_id=test_patient_id, 
        modality=sample_modality, 
        diagnosis=sample_diagnosis, 
        confidence=sample_confidence, 
        llm_report=sample_llm_text
    )

    print("\n[DEBUG] Generated FHIR R4 JSON Payload:")
    print(json.dumps(fhir_payload, indent=2))

    # 4. Transmit
    bridge.transmit_to_emr(fhir_payload)
