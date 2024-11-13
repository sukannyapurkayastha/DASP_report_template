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
- **Update 3**:
- **Challenges**:
    - Design of the testset
        - What information should it contain and how the sentences should be preprocessed
        - In general, the preprocessing with spacy was very difficult. But I think also with an LLM it would be
          difficult.
- **Next Steps**:
    - Write tests for the OpenReviewLoader
    - Extend the functionality of the OpenReviewLoader (if needed)
    - API interface between Backend and Frontend

### Week [n-2]

- **Update 1**:
- **Update 2**:
- **Update 3**:
- **Challenges**:
- **Next Steps**:

---
