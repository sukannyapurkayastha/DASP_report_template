
We have currently 5 components:
- frontend
- backend (preprocess, segmentation and score calculation)
- attitude_classifier
- request_classifier
- summary_generator

#### **3.1 Install Environment**

Folders have same name as components. In each folder there is a requirements file and conda environment file. 
- usage of conda environment file: It contains all information for environment like python verison, list of libs and versions of libs. It's recommended to use this file firstly, if it omits error, try requirements.txt file.
- usage of requirements.txt: It's served for docker. But you can also install your environment with this file. Python 3.10 is used for all containers. Versions of libs are not specified in requirements file, in order to prevent version mismatching errors.

#### **3.2 Run Services From Terminal**
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