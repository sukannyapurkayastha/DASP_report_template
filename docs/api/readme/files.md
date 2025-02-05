

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
  - `tests`: tests for the frontend

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
  - `summary`: includes scripts for training of summary generation models. Specifically training of T5-large, BART-large and Llama2.
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
  - `models/llama2`: contains llama2 model


- **Files**:
  - `main.py`: creates FastAPI app, runs it and defines a FastAPI endpoint and executes the actual prediction by making each prediction step by step and structures processed data into a list that is turned into a dictonaries
  - `data_processing.py`: Generates structured input text for llama2 from overview, attitude, and request DataFrames.
  - `input_to_pompt_converter.py`: restructures the input from the models into a prompts for the LLM
  - `predict_LLAMA2.py`: makes a prediction for the given prompt
  - `requirements.txt`: contains packages used to execute prediction. Can be used especially for exection on windows
  - `summary_env.yml`: contains the packages and dependencies for the summary generator when running on linux/slurm servers
  - `slurm_test.py`: test function for slurm
  