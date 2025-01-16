# [Student's Name] Weekly Reports

## Student Information
- **Name**: Zhijingshui
- **Project Title**: Review Overview Generation for papers through Attitude root-Review request Mapping
- **Mentor**: Sukannya Purkayastha
- **Course**: [	Data Analysis Software Project Seminar for Natural Language/6,0]

---

## Reports

### Week 1
- **Update 1**: [papers_summary](paper_summary/papers_summary.md)
- **Update 2**: fill out IP form
- **Update 3**: check how to get reviews for ICLR 2024 OpenReview. 
  - https://docs.openreview.net/how-to-guides/data-retrieval-and-modification/how-to-export-all-reviews-into-a-csv
  - https://docs.openreview.net/how-to-guides/data-retrieval-and-modification/how-to-get-all-reviews#venues-using-api-v2
- **Challenges**:
  ### questions about tasks
  
  > Identify **important concerns** from multiple reviews via Attitude Root Taxonomy [[1](https://aclanthology.org/2023.emnlp-main.894.pdf)]
  > 
  > 1. Finetuning classifiers on the JitsuPEER Dataset and transfer to our scenario.
  - so only attitute root? no attitute themes? if so we should use DISAPERE? JitsuPEER Just used attitute root labels from DISAPERE
      
      ```
      ├── Arg_other
      ├── Asp_clarity
      ├── Asp_meaningful-comparison
      ├── Asp_motivation-impact
      ├── Asp_originality
      ├── Asp_replicability
      ├── Asp_substance
      ```
      
  - if we need themesm where is the model and code piece for predicting themes? 
  > Link the **concerns to requests** from the reviewers eg., all reviewers want more baselines using Requests from DISAPERE taxonomy [[2](https://arxiv.org/pdf/2110.08520)]
  > 
  > 1. Finetuning classifiers on the DISAPERE dataset and inference on the identified concerns.
  - same model? For the six classification tasks, we use bert-base (Devlin et al., 2019) to pro-duce sentence embeddings for each sentence, then classify the representation of the [CLS] token using a feedforward network.
  - what we need to do? collect data from ICLR 2021? annotate some of them? finetune model and use the model to annotate the rest of datasets?
  or use the model from paper, predict for all the data and check if it’s correct
  - what kind of request labels do we have, like following?
      ![alt text](image.png)
- **Next Steps**: 
  - create scrum boards
  - collect reviews using OpenReview API
  - experiment with building website

### Week [2]

- **Update 1**: 
- **Update 2**: 
- **Update 3**: 
- **Challenges**: libs install for jitsupeer repo not successful, it causes error even with requirements.txt
- **Next Steps**: 

### Week [3]

- **Update 1**: review-to-description model runs successfully on slurm server with rouge score
- **Update 2**: experiment attitute theme classifier
- **Update 3**: 
- **Challenges**: 
- **Next Steps**: finish attitute theme classifier; start to connect model to frontend

### Week [4]

- **Update 1**: connect attitude roots to frontend with fastapi
- **Update 2**: build up attitute theme classifier
- **Update 3**: evaluation review-to-description model with embedding similarity and edit similarity
- **Challenges**: decide which infrastructure to use: 1. frontend backend mix up style; 2. connect model with api call
- **Next Steps**: improve attitute theme classifier; connect attitue theme to frontend

### Week [5]

- **Update 1**: connect attitude themes to frontend with fastapi
- **Update 2**: check data leakage of attitude themes classifier, analyse dataset and results of attitude themes classifier
- **Update 3**: 
- **Challenges**: train-test-val data split for multilabel scenario
- **Next Steps**: improve attitute theme classifier; merge branches
---
