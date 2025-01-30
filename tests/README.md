# Integration tests
This is a module for integration tests, given that our project structure is quite divided, normal integration tests are 
quite intricate (having to start multiple servers). Furthermore, it is not possible to only test a part of the 
pipeline since the backend server works as an orchestrator and makes multiple further calls to our LLM models.

## How to run
**1. Set Up the Model Environment**

    conda env create -f integration_tests_env.yaml
    conda activate DASP_integration
    python -m spacy download en_core_web_sm


**2. Run the tests**

For openreview we need your username and password to connect to the API. You can pass as it as arguments in the 
command line by replacing YOUR_USERNAME with your actual openreview username and YOUR_PASSWORD with your openreview
password.

Please run this from the root folder of the project to ensure correct module imports.

    pytest .\tests\test_openreview_textprocessor_integration.py --username="YOUR_USERNAME" --password="YOUR_PASSWORD"
    pytest .\tests\test_uploadedfileprocessor_textprocessor_integration.py

For the following test, first start the attitude_classify model on port 8002:

    pytest .\tests\test_textprocessor_classify_attitude_integration.py --username="YOUR_USERNAME" --password="YOUR_PASSWORD"


