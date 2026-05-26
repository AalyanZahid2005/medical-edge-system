# M3 Network Map

## Services

| Component | Host | Port | Protocol | Purpose |
|---|---|---:|---|---|
| Flask API Server | 127.0.0.1 / Jetson IP | 5000 | HTTP/WebSocket | API endpoints: /analyze, /queue, /correct, /federated_sync, /stream |
| Orthanc PACS REST | 127.0.0.1 / PACS Laptop | 8042 | HTTP REST | Orthanc REST API: /system, /studies, /instances |
| Orthanc DICOM | 127.0.0.1 / PACS Laptop | 4242 | DICOM | DICOM C-FIND / C-STORE |
| SQLite Database | Local Project Folder | file | SQLite | fyp_medical.db |
| DICOM C-FIND Poller | Local/Jetson | script | Python | Polls Orthanc and downloads new scans |
| DICOM Listener | Local/Jetson | configured port | DICOM | Receives incoming DICOM studies |

## Verified Working Flow

Orthanc PACS -> dicom_cfind.py -> downloaded DICOM -> api_server.py /analyze -> SQLite DB -> confidence routing -> active learning queue -> DICOM SR log -> /correct -> EWC -> /federated_sync.

## Verified Evidence

- Orthanc /system works.
- Orthanc /studies and /instances return IDs.
- dicom_cfind.py found and routed 1 fresh DICOM.
- /analyze created inference records.
- dicom_sr_log contains SENT rows.
- network_stress_test.py completed 50/50 successful requests.
- Final DB after stress test:
  - Studies: 77
  - Inference Records: 70
  - DICOM SR Logs: 53
