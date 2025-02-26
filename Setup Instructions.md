# **Review Overview Generation**
Here is the link to our project git repo: https://github.com/sukannyapurkayastha/DASP_report_template
## **Installation and Setup Instructions**

This guide helps you set up and run the project, which consists of 2 main parts:
- **Deployment Guide**
- **Developer Guide**

#### **Clone the Repository** 

      git clone https://github.com/sukannyapurkayastha/DASP_report_template.git
      cd your-project
---
### 1. Deployment Guide

#### **Prerequisites**

- **Git** for cloning the repository.
- **Docker** for containerizing and running the application.
- **Docker Compose** for managing multi-container environments.
- **NVIDIA Container Toolkit** enables GPU acceleration with nvidia/cuda images.
[Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

#### **Application Deployment**

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

### **2. Developer Guide**
This is the instruction for developers, who want to debug or directly work with our codes.
We have currently 5 components:
- frontend
- backend (preprocess, segmentation and score calculation)
- attitude_classifier
- request_classifier
- summary_generator

#### **Install Environment**

Folders have same name as components. In each folder there is a requirements file and conda environment file. 
- usage of conda environment file: It contains all information for environment like python verison, list of libs and versions of libs. It's recommended to use this file firstly, if it omits error, try requirements.txt file.
- usage of requirements.txt: It's served for docker. But you can also install your environment with this file. Python 3.10 is used for all containers. Versions of libs are not specified in requirements file, in order to prevent version mismatching errors.

#### **Run Services From Terminal**
```bash
cd frontend
streamlit run app.py
```
Following command is valid for all these 4 components, replace {component_folder} with backend, attitude_classifier, request_classifier, summary_generator
```bash
cd {component_folder}
python main.py
```
After running all 5 commands, the web application is running on localhost:8000. Note that running without docker, it's available at port 8000.
