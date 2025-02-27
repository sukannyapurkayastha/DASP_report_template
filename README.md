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

This guide helps you set up and run the project, which consists of 2 main parts:
- **Deployment**
- **Setup Instructions for Developer**

#### **Clone the Repository** 

      git clone https://github.com/sukannyapurkayastha/DASP_report_template.git
      cd your-project
---
### 2.1 Deployment

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

### **2.2 Developer Guide**
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

## **3. Architecture**


##### **3.1 Frontend**
The Frontend is the user interface of the system where individuals log in, provide a URL to OpenReview, and optionally download and then upload the filled out templates. The Frontend handles interactions, collects the user’s input (including files and URLs), and displays the resulting classification output once the Backend has processed everything.

##### **3.2 Backend**
Once the Frontend submits data (whether uploaded files or URLs), the Backend starts analyzing the provided data. It first performs formatting and segmentation, breaking the reviews into sentences. From there, the system routes the segments to various prediction modules. The “Request Prediction” module handles general categorization of the Request, while an “Attitude/Theme Prediction” module determines attitude and corresponding themes and descriptions. After processing these steps, the Backend compiles the outputs—now in the form of classified sentences or structured results—and sends them back to the Frontend to display to the user.

##### **3.3 Model Training**
As part of our framework, there is the model training. Initially we trained the models used in the Backend to perform the neccesary classification tasks. This training process results in Model Artifacts, such as updated model parameters, which the Backend uses during its prediction steps. If necessary, the existing model and code files can be used to update and improve existing models with new data or better models.

*Request Classifier*

For the Request Classifier we used two fine tuned models to firstly label what sentences are requests and then have a fine grained classification for the type of request. The first is a binary classifier which achieved an F1 Score of 91%, we tried oversampling, but the results were worse. For the fine requests we have a multi class prediction for which we also tried different models and approaches. The best results were with the addition of thresholding resulting in a F1 score of 62%.

*Attitude and Theme Classifier*

For the Attitude Roots which represent the underlying believes we made a multi class prediction and tried different models and approaches (normal training, oversampling and hybrid with normal training and oversampling). We achieved an F1 Score of 62% with the BERT model which was pretrained on domain knowledge.


*Summary Generation*

To create the summary—which aggregates all results sorted by “Overview”, “Attitude Roots” and “Request Information”—we tried different models and evaluated their performance using the BERT score. More specifically, we pre-structured the summary using Python and then generated predictions only for the collections of comments corresponding to a particular attitude root or request, as determined by our other models.

Data Collection and Labeling Process:
We manually collected data from nine OpenReview threads to ensure a balanced distribution of overall ratings. Specifically, the selection includes three examples from each rating category: "low" with overall score < 4, "average" with overall score >=4 but < 7 and "high" with overall score >= 7. Instead of treating each individual comment as a separate data point, we clustered sentences so that each cluster represents the set of comments associated with a single paper and corresponds to a specific "attitude root" or request as identified by our preliminary models (e.g., all comments complaining about a typo). This clustering resulted in 174 aggregated data points (see model_training/nlp/summary/data/real_world_data_unlabeled.jsonl).
Next, each data point was labeled using OpenAI's so far most capable model ChatGPT o1. We then proofread these labels to ensure high quality (see model_training/nlp/summary/data/real_world_data_labeled.jsonl).

Prediction:
We tried sequence to sequence approaches with BART-large and T5 using an 80%-10%-10% train-validation-test split.
The best results were obtained using Llama2 with a 10-shot prompt, achieving an F1-BERT score of 69% on the test data.

##### **3.4 Communication Flow**

*Frontend → Backend*

The Frontend issues secure API calls to the Backend when users log in, provide URLs, or upload filled templates.
The Backend processes these incoming requests—formatting and segmenting the data—and routes them to the appropriate prediction modules.
The Backend functions as a API gateway.

*Backend → Frontend*

Once the predictions are complete, the Backend responds via API calls back to the Frontend, delivering classified sentences, sentiment results, or other structured outputs.
The Frontend then displays these results to the user in a clear, readable format.

*Model Training and Backend*

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
  - `main.py`: defines a FastAPI endpoint, creates FastAPI app and runs it
  - `model_prediction.py`:  structures, classifies and transforms processed data into target table
  - `description_generation.py`: generates description for class clusters
  - `attitude_classifier_env.yaml`: contains the packages and dependencies
  - `attitude_desc.csv`: contains mapping information between attitude clusters and description

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

#### **4.6 `summary_generator`**

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
  
## **5 Data**

For this project, we utilized two primary datasets:

*JiujitsuPeer Dataset*

Used for developing the attitude and theme prediction models.
The dataset comes pre-labeled by its original creators, providing ground truth annotations for sentiment or stance (attitude) and thematic categories (theme).
We selectively extracted and subdivided only those sections most relevant for our model objectives, ensuring training data remained highly focused on the classification tasks at hand.

*DISAPERE Dataset*

Used for building and refining our request prediction models.
Like the JiujitsuPeer Dataset, DISAPERE was pre-labeled with relevant review actions and requests, allowing us to apply segmentation and filtering specific to the request-classification requirements.
We further tailored the dataset by removing or restructuring fields not pertinent to predicting review actions, simplifying integration with our overall pipeline.

By leveraging these pre-labeled datasets—and performing only minimal pre-processing to isolate the pertinent fields—we streamlined the model training phase while retaining high-quality annotations for the core prediction tasks.

## **5 Testing**

We adopted a multi-level testing strategy to ensure both reliability and maintainability across our application:

*Unit Tests*

Implemented extensively for backend modules and individual model components.
Validate each function's correctness, focusing on data loading, preprocessing methods, and core inference logic in isolation.

*Integration Tests*

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
