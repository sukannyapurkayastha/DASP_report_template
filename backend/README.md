## DASP Backend
1. Create conda environment with "conda env create -f backend_env.yaml"
2. Install spacy model with "python -m spacy download en"

3. activate backend service:
``` python main.py ```
and you should see something like:
```
INFO:     Started server process [30087]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```


4. run frontend
``` streamlit run app.py ```