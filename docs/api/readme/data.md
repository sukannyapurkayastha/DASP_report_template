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