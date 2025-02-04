---
## **2. Installation and Setup Instructions**

This guide helps you set up and run the project, which consists of three main parts:
- **Model Training (NLP models environment)**
- **Backend (APIs connecting frontend and models)**
- **Frontend (UI environment via streamlit)**

---

#### **Prerequisites**

- **Git** for cloning the repository.
- **Python 3.12.7+** installed on your system.
- **conda**  for managing Python environments.

---
#### **1. Clone the Repository** 

      git clone https://github.com/sukannyapurkayastha/DASP_report_template.git
      cd your-project

**2. Set Up the Model Environment**

      pip install requirements.txt

      conda env create -f frontend_env.yaml
      conda activate frontend_env.yaml
      conda env create -f backend_env.yaml
      conda activate backend_env.yaml
      conda env create -f attitude_classifier_env.yaml
      conda activate attitude_classifier_env.yaml
      conda env create -f request_classifier_env.yaml
      conda activate request_classifier_env.yaml
      conda env create -f summary_env.yaml
      conda activate summary_env.yaml

**3. Start the application**

      streamlit app.py