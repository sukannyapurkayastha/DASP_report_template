<p align="center">
  <img src="logo.png" width="200" alt="Logo 1" style="margin-right: 20px;">
  <img src="logo_header.png" width="200" alt="Logo 2">
</p>

# **Review Overview Generation**

## **1.Project Overview**

At **Paper Review Aggregator**, our mission is to simplify the work of meta-reviewers by providing an AI-driven platform that aggregates and summarizes paper reviews. By reducing manual effort and enhancing decision-making, we aim to streamline the peer-review process for academic research.
This application was developed as part of the **"Data Analysis Software Project"** course at **Technische Universität Darmstadt, Germany**. The project allowed Master’s students to apply their data analysis, software engineering, and machine learning skills to solve real-world problems.

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


## **3. Architecture and Design Notes**
#### **3.1 Architecture**

![alt text](image.png)

#### **3.2 Design Notes**

##### **3.2.1 Frontend**
The Frontend is the user interface of the system where individuals log in, provide a URL to OpenReview, and optionally download and then upload the filled out templates. The Frontend handles interactions, collects the user’s input (including files and URLs), and displays the resulting classification output once the Backend has processed everything.

##### **3.2.2 Backend**
Once the Frontend submits data (whether uploaded files or URLs), the Backend starts analyzing the provided data. It first performs formatting and segmentation, breaking the reviews into sentences. From there, the system routes the segments to various prediction modules. The “Request Prediction” module handles general categorization of the Request, while an “Attitude/Theme Prediction” module determines attitude and corresponding themes and descriptions. After processing these steps, the Backend compiles the outputs—now in the form of classified sentences or structured results—and sends them back to the Frontend to display to the user.

##### **3.2.3 Model Training**
As part of our framework, there is the model training. Initially we trained the models used in the Backend to perform the neccesary classification tasks. This training process results in Model Artifacts, such as updated model parameters, which the Backend uses during its prediction steps. If necessary, the existing model and code files can be used to update and improve existing models with new data or better models.

##### **3.2.3 Communication Flow**
Frontend → Backend

The Frontend issues secure API calls to the Backend when users log in, provide URLs, or upload filled templates.
The Backend processes these incoming requests—formatting and segmenting the data—and routes them to the appropriate prediction modules.
The Backend functions as a API gateway.

Backend → Frontend

Once the predictions are complete, the Backend responds via API calls back to the Frontend, delivering classified sentences, sentiment results, or other structured outputs.
The Frontend then displays these results to the user in a clear, readable format.

Model Training and Backend:

The models trained in the Backend are stored in the designated containers with are activated when the Backend is called.




## **4. Detailed description of the files**

#### **4.1 `backend`**

The backend is responsible for data collection from OpenReview, preprocessing the data, and managing the workflow between the models and the frontend. It sends preprocessed data to the models for processing, retrieves the results, and communicates the processed data to the frontend for further use.

- **Subdirectories**:
  - `dataloading`: includes scripts for using the open review API
  - `text_processing`: includes file and scripts which are used to preprocess and prepare review for the models
  - `test`: includes test for the text processing

- **Files**:
  - `data_processing.py`: calls functions from text processing  
  - `main.py`: creates FastAPI app and runs it
  - `routers.py`: defines a FastAPI endpoint and structures processed data into dictonaries
  - `backend_env.yaml`: contains the packages and dependencies for the backend

---

#### **4.2 `frontend`**

The front end of this application is the user-facing interface built with Streamlit and related UI components. It is responsible for displaying the overview over the given reviews, receive the data in form of URL or file, and providing an interactive experience within the application. It is connected via API calls to the backend and includes several features of displaying the data for the user.

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

#### **4.3 `model_training`**

The model training folder includes the scripts and data which which the models where trained.

- **Subdirectories**:
  - `attitude_classifier`: includes the scripts for fine tuning the a model for multi class classification for attitude roots
  - `e2e_review_to_desc`:  includes the scripts and the data for fine tuning the a model for multi class classification for
                           attitude themes and matches them to a description 
  - `request_classifier`: includes the data (DISAPERE) and the scripts for binary classication for Review_Action Request vs. All 
                          and a multi class classification for the fine review actions 
  - `review_to_theme`: includes scripts for mapping review sentences to themes
  - `summary`: includes scripts for summary generation models. Specifically training of T5-large, BART-large and Llama2.
---

#### **4.4 `attitude_classifier`**

The attitude classifier can contains the model and contains the scripts for the request classifier pipeline.

- **Files**:
  - `main.py`: creates FastAPI app and runs it
  - `routers.py`: defines a FastAPI endpoint and structures processed data into dictonaries
  - `backend_env.yaml`: contains the packages and dependencies for the backend
  - `attitude_classifier.yaml`: contains the packages and dependencies for the enviroment

---

#### **4.5 `request_classifier`**

The request classifier can contains the model and contains the scripts for the request classifier pipeline.

- **Subdirectories**:
  - `classification`: includes scripts for using the request classifier pipeline


- **Files**:
  - `main.py`: creates FastAPI app and runs it
  - `routers.py`: defines a FastAPI endpoint and structures processed data into dictonaries
  - `request_classifier.yaml`: contains the packages and dependencies for the enviroment

---

#### **4.5 `summary_generator`**

The summyry generator folder can contains the model and contains the scripts for the summary generations.

- **Subdirectories**:
  - `models/llama2`: includes llama2 model


- **Files**:
  - `main.py`: creates FastAPI app, runs it and defines a FastAPI endpoint and executes the actual prediction by making each prediction step by step and structures processed data into a list that is turned into a dictonaries
  - `data_processing.py`: Generates structured input text for llama2 from overview, attitude, and request DataFrames.
  - `input_to_pompt_converter.py`: converts a given string into a few-shot-prompt
  - `predict_LLAMA2.py`: makes a prediction for the given prompt
  - `requirements.txt`: contains packages used to execute prediction. Can be used especially for exection on windows
  - `summary_env.yml`: contains the packages and dependencies for the summary generator when running on linux/slurm servers
  
## **5 Data**

For this project, we utilized two primary datasets:

JiujitsuPeer Dataset

Used for developing the attitude and theme prediction models.
The dataset comes pre-labeled by its original creators, providing ground truth annotations for sentiment or stance (attitude) and thematic categories (theme).
We selectively extracted and subdivided only those sections most relevant for our model objectives, ensuring training data remained highly focused on the classification tasks at hand.

DISAPERE Dataset

Used for building and refining our request prediction models.
Like the JiujitsuPeer Dataset, DISAPERE was pre-labeled with relevant review actions and requests, allowing us to apply segmentation and filtering specific to the request-classification requirements.
We further tailored the dataset by removing or restructuring fields not pertinent to predicting review actions, simplifying integration with our overall pipeline.

By leveraging these pre-labeled datasets—and performing only minimal pre-processing to isolate the pertinent fields—we streamlined the model training phase while retaining high-quality annotations for the core prediction tasks.

## **5 Testing**

We adopted a multi-level testing strategy to ensure both reliability and maintainability across our application:

    Unit Tests
        Implemented extensively for backend modules and individual model components.
        Validate each function's correctness, focusing on data loading, preprocessing methods, and core inference logic in isolation.

    Integration Tests
        Primarily target the frontend and its interactions with the backend APIs.
        Assess end-to-end functionality—verifying that data flows correctly from the user interface, through the backend services, and back again with the expected responses and outputs.


## **5 Contact**

If you have any questions or suggestions regarding this project, feel free to reach out:

- **Your Name**: Johannes Lemken
- **Email**: johannes.lemken@stud.tu-darmstadt.de
- **GitHub Profile**: Johannes Lemken (https://github.com/JohannesLemken)

Thank you for your interest in this project!

## **6 Contributions**

In this project participated Johannes Lemken, Carmen Appelt, Zhijingshui Yang, Philipp Oehler and Jan Werth.
We were supervised by Sukannya Purkayastha.