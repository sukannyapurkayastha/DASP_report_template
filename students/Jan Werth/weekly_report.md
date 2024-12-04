# [Jan Werth] Weekly Reports

## Student Information

- **Name**: Jan Werth
- **Project Title**: Review Overview Generation for papers through Attitude root-Review request Mapping
- **Mentor**: Sukannya Purkayastha
- **Course**: Data Analysis Software Project Seminar for Natural Language (WiSe 2024/25) / 6 ECTs

---

## Reports

### Week [1]

- **Update 1**: Read literature for JITSUPEER and DISAPEER dataset.
- **Update 2**: Tried the openreview api -> Was able to download all the reviews for an article if ID is provided.
- **Update 3**: Tried streamlit and watched a YT tutorial.
- **Challenges**:
- **Next Steps**:

### Week [2]

- **Update 1**: Created a Wrapper class OpenReviewLoader for the OpenReview API v2. I tried to make it openly designed
  as somewhat
  possible so that we can extend it later to also get some other information from the api regarding papers (paper
  metadata, etc. which we can use in the frontend)
- **Update 2**: The largest part of the OpenReviewLoader until yet is the possibility to create a test set, which was my
  task for this week.
- **Update 3**: Setting up slurm
- **Challenges**:
    - Design of the testset
        - What information should it contain and how the sentences should be preprocessed
        - In general, the preprocessing with spacy was very difficult. But I think also with an LLM it would be
          difficult.
    - How to install python/conda on slurm? How do I change node?
- **Next Steps**:
    - Write tests for the OpenReviewLoader
    - Extend the functionality of the OpenReviewLoader (if needed)
    - API interface between Backend and Frontend

### Week [3]

- **Update 1**: Refactoring of OpenReviewLoader with better spacy sentence segmentation. It also works for papers from
  Neurips 2024
- **Update 2**:
- **Update 3**:
- **Challenges**: 
  - Handling text data (it's no fun) and sentence segmentation. Even though you can do a lot with spacy
    there are so many special cases you have to account for.
- **Next Steps**:
  - Reading reviews from files/manual upload (how to handle that if they are not in the same format as the api response?)

### Week [4]

- **Update 1**: Refactoring of TextProcessor to split sentences at linebreak
- **Update 2**: Template and read from template
- **Update 3**: Login and provide url functionality to frontend. Provide template, upload template and read from word file
- **Challenges**:
  - Reading from word files is really complicated
  - Backend and frontend connection
- **Next Steps**:
  - Better connection between backend and frontend
  - Expanding the review functionality for other venues/conferences

### Week [n-2]

- **Update 1**:
- **Update 2**:
- **Update 3**:
- **Challenges**:
- **Next Steps**:

---
