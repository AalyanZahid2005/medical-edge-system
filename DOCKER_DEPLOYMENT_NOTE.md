# M3 Docker Deployment Note

## Docker Compose File

The file `docker-compose.m3.yml` has been prepared for M3 deployment. It defines:

- Orthanc PACS container
- Medical AI Flask API container
- Port 8042 for Orthanc REST
- Port 4242 for Orthanc DICOM
- Port 5000 for Flask API

## Current System Limitation

Docker runtime test could not be performed on this Windows machine because Docker is not installed or not available in PATH.

PowerShell error:

`docker: The term 'docker' is not recognized`

## Verified Local Deployment

Although Docker runtime is pending, the same services were verified locally:

- Orthanc REST API works on port 8042.
- Orthanc DICOM studies and instances are visible.
- Flask API works on port 5000.
- SQLite database is working.
- C-FIND fresh DICOM routing was tested.
- `/analyze`, `/fhir_report`, `/analytics`, `/billing_summary`, `/queue`, `/correct`, and `/federated_sync` work.
- DICOM SR logs contain SENT records.

## Production Requirement

For final hospital deployment, install Docker Desktop or Docker Engine and run:

`docker compose -f docker-compose.m3.yml up -d`
