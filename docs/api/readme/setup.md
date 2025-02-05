

This guide helps you set up and run the project, which consists of three main parts:
- **Model Training (NLP models environment)**
- **Backend (APIs connecting frontend and models)**
- **Frontend (UI environment via streamlit)**

---

#### **Prerequisites**

- **Git** for cloning the repository.
- **Docker** for containerizing and running the application.
- **Docker Compose** for managing multi-container environments.
- **NVIDIA Container Toolkit** enables GPU acceleration with nvidia/cuda images.
[Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

---
#### **1. Clone the Repository** 

      git clone https://github.com/sukannyapurkayastha/DASP_report_template.git
      cd your-project

#### **2. Application Deployment**

The application runs as a set of Docker containers orchestrated with Docker Compose. To start the app detached in your local device simply run:
```bash
docker compose up -d
```
Check logs for a specific service (current status or error message):
```bash
docker compose logs -f <service_name>
```
for example <service_name> could be: request_classifier

If you are hosting the application on your local device, by default the website is published at port 80 on your local machine. For deployment it's highly recommanded to put the application behind a webserver like Caddy, Apache or Nginx.

The application is available online at **[https://reviewoverview.ukp.informatik.tu-darmstadt.de](https://reviewoverview.ukp.informatik.tu-darmstadt.de)**, if you have access to UKP or HRZ VPN. Since the proxy handles SSL termination, currently we don't have webserver in front of Streamlit.