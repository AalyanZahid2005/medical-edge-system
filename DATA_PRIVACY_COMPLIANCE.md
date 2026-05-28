# Data Privacy Compliance Document — M3

## 1. System Overview

This project is a local medical AI edge system. It processes DICOM medical scans through Orthanc PACS, Flask API, SQLite database, and DICOM SR logging without requiring cloud upload.

## 2. Local Processing

All inference and routing steps are performed locally on the edge/laptop environment. The system does not require patient images to be uploaded to any cloud server for the current demo.

## 3. Local Storage

The following data is stored locally in SQLite:

- Patient ID
- Study UID
- Modality
- Predicted class
- Confidence and uncertainty
- Routing tier
- Active learning status
- DICOM SR log status
- Billing/ICD-10 summary

## 4. Orthanc PACS Privacy

Orthanc runs inside the local hospital/project network. The system uses local REST/DICOM communication through:

- Orthanc REST: port 8042
- Orthanc DICOM: port 4242
- Flask API: port 5000

## 5. HIPAA Alignment

The design supports HIPAA-style privacy principles because:

- Medical data remains local.
- No cloud transmission is required.
- Access can be restricted to hospital LAN.
- DICOM records and AI reports are stored locally.
- Future deployment can add HTTPS and role-based authentication.

## 6. GDPR Alignment

The design supports GDPR principles by using:

- Data minimization
- Local processing
- Limited patient identifiers
- Local auditability
- No external sharing by default

## 7. Pakistan PDPA Alignment

The system supports Pakistan data protection expectations by keeping patient data inside the local hospital/project network and avoiding external cloud transfer.

## 8. Current Demo Limitations

Current demo mode uses local HTTP. Production deployment should add:

- HTTPS reverse proxy
- JWT enforcement on all API endpoints
- User role management
- Encrypted database storage
- Full audit logs

## 9. Verified M3 Evidence

- API /health works.
- Orthanc /system works.
- SQLite database exists.
- DICOM SR logs are stored.
- C-FIND workflow was tested with Orthanc.
- FHIR-style report endpoint works.
- Analytics and billing endpoints work.

## 10. Conclusion

The M3 networking design is suitable for local hospital-network deployment as a prototype. For production, HTTPS, JWT, formal access control, and encrypted storage should be enabled.
