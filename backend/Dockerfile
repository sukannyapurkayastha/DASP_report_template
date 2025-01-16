FROM python:3.10-slim

# Set the working directory
WORKDIR /backend

# copy requirements file firstly, so it saves time if rebuilt.
COPY ./requirements.txt .
# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt
RUN python3 -m spacy download en_core_web_sm

# Copy the Streamlit app
COPY . .

# Step 6: Expose the port the FastAPI app will run on
EXPOSE 8080

# Step 7: Set the entry point to run the FastAPI app with Uvicorn
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]


