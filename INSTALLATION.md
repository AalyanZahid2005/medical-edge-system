# Hospital IT Deployment Guide

## Prerequisites
* NVIDIA Jetson Nano (Flashed with JetPack 5.0+).
* A local Hospital PC running Orthanc PACS.
* Docker and Docker Compose installed on both machines.
* Static IP addresses assigned to both the Jetson and the PACS PC.

## Step 1: Deploy the AI Edge Server (Jetson Nano)
1. SSH into the Jetson Nano.
2. Clone the production repository:
   `git clone https://github.com/your-hospital-repo/medical-edge-system.git`
3. Navigate to the deployment folder:
   `cd medical-edge-system/deploy`
4. Spin up the containers:
   `docker-compose up -d`
*(Note: This starts the `medai-inference`, `medai-admin`, and `medai-dicom-listener` services).*

## Step 2: Configure DICOM Routing (Orthanc PC)
1. Open the Orthanc Configuration file (`orthanc.json`).
2. Add the Jetson Nano to the `DicomModalities` list:
   ```json
   "JetsonAI" : [ "MED_AI", "192.168.1.100", 4242 ]