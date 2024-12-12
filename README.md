<p  align="center">
  <img src='logo.png' width='200'>
</p>

# **Review Overview Generation**

## **1.Project Overview**

At **Paper Review Aggregator**, our mission is to simplify the work of meta-reviewers by providing an AI-driven platform that aggregates and summarizes paper reviews. By reducing manual effort and enhancing decision-making, we aim to streamline the peer-review process for academic research.
This application was developed as part of the **"Data Analysis Software Project"** course at **Technische Universität Darmstadt, Germany**. The project allowed Master’s students to apply their data analysis, software engineering, and machine learning skills to solve real-world problems.

---
### **2. Installation and Setup Instructions**

This guide helps you set up and run the project, which consists of three main parts:
- **Models (NLP models environment)**
- **Backend (APIs connecting frontend and models )**
- **Frontend (UI environment via streamlit)**

---

**Prerequisites**

- **Git** for cloning the repository.
- **Python 3.12.7+** installed on your system.
- **conda**  for managing Python environments.

---
1. **Clone the Repository**  
   git clone https://github.com/your-username/your-project.git
   cd your-project

2. **Set Up the Model Environment**
- conda env create -f model_env.yaml
  conda activate your-model-env
- conda env create -f backend_env.yaml
  conda activate your-backend-env
- conda env create -f frontend_env.yaml
  conda activate your-frontend-env

3. **Start the application**
  Run the application with streamlit app.py

Architecture and Design Notes
##### **3. Architecture and Design Notes**

![alt text](image.png)

Design notes


##### **4. Detailed description of the files**

##### **4.1 `backend`**

The backend is responsible for data collection from OpenReview, preprocessing the data, and managing the workflow between the models and the frontend. It sends preprocessed data to the models for processing, retrieves the results, and communicates the processed data to the frontend for further use.

- **Subdirectories**:
  - `dataloading`: includes scripts for using the open review API
  - `models`: includes the models which are used to predict the NLP component of this application
  - `nlp`: ...
  - `text_processing`: includes file and scripts which are used to preprocess and prepare review for the models

- **Files**:
  - `__init__.py`: ...
  - `backend_env.yaml`: includes specific enviroment for the backend functionality
  - `data_processing.py`: calls functions from text processing  
  - `main.py`: creates FastAPI app and runs it
  - `routers.py`: defines a FastAPI endpoint and structures processed data into dictonaries

---

##### **4.2 `frontend`**

Thies front end of this application is the user-facing interface built with Streamlit and related UI components. It is responsible for displaying the overview over the given reviews, receive the data in form of URL or file, and providing an interactive experience within the application. It is connected via API calls to the backend and includes several features of displaying the data for the user.

- **Subdirectories**:
  - `images`: includes the images needed for displaying the front end 
  - `data`: includes the templates for the user if data of reviews is manually input 
  - `modules`: includes several scriptsfor features of the frontend like containers for specific parts of the data or elements 
              of the website like the side bar
  - `clients`: includes the client for excessing OpenReview
  - `.streamlit`: includes a configfile for the custom design

- **Files**:
  - `__init__.py`: Initializes the Python package and sets up the module structure (not in use)
  - `about.py`: calls about under modules which contains content and layout logic for the "About" page of the front end
  - `app.py`: Coordinates the overall structure and routing of the front end application’s pages
  - `contact.py`: calls about under modules which  implements the "Contact" page, providing forms or information for user inquiries
  - `data.ipynb`: jupyper notebook with specifies the right structure and content of the dataframes for the front end 
  - `frontend_env.yaml`: Defines the environment dependencies and configurations for running the front end
  - `home_page.py`: Implements the "Home" page, showcasing primary content and user entry points
  - `landing_page.py`: Sets up the initial landing view, guiding users to the main sections of the application
  - `main_page.py`: Acts as the core page aggregating or linking to the main functionalities of the site
  - `README`: A documentation file describing the purpose, setup, and usage of the front end
  - `run.py`: A script that launches the Streamlit application and makes the front end accessible
  - `streamlit_app.py`: The primary Streamlit script coordinating UI components, page navigation, and interactions

---

##### **4.3 `model_training`**

The model training folder includes the scripts and data which which the models where trained.

- **Subdirectories**:
  - `attitude_classifier`: includes the scripts for fine tuning the a model for multi class classification for attitude roots
  - `e2e_review_to_desc`:  includes the scripts and the data for fine tuning the a model for multi class classification for#
                           attitude themes and matches them to a description 
  - `request_classifier`: includes the data (DISAPERE) and the scripts for binary classication for Review_Action Request vs. All 
                          and a multi class classification for the fine review actions 
  - `review_to_theme`: includes scripts for mapping review sentences to themes
---