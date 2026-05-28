import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "http://127.0.0.1:5000/analyze"
N = 50

def send_one(i):
    t0 = time.time()
    data = {
        "patient_id": f"PT-STRESS-{i:03d}",
        "modality": "brain_mri",
        "study_uid": f"STUDY-STRESS-{i:03d}",
        "precision": "fp16"
    }

    try:
        r = requests.post(URL, data=data, timeout=20)
        return i, r.status_code, round((time.time() - t0) * 1000, 2)
    except Exception as e:
        return i, "ERROR", str(e)

t0 = time.time()
results = []

with ThreadPoolExecutor(max_workers=10) as ex:
    futures = [ex.submit(send_one, i) for i in range(1, N + 1)]
    for f in as_completed(futures):
        result = f.result()
        results.append(result)
        print(result)

ok = sum(1 for _, status, _ in results if status == 200)
total_ms = round((time.time() - t0) * 1000, 2)

print("\n=== NETWORK STRESS SUMMARY ===")
print("Total requests:", N)
print("Successful:", ok)
print("Failed:", N - ok)
print("Total time ms:", total_ms)
print("Average ms/request:", round(total_ms / N, 2))