import argparse
import pandas as pd

from loaders.openreview_loader import OpenReviewLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("username", help="Openreview username")
    parser.add_argument("password", help="Openreview password")

    args = parser.parse_args()

    try:
        client = OpenReviewLoader(username=args.username, password=args.password)
    except Exception as e:
        print(f"{e.args[0]["status"]}: {str(e.args[0]["message"])}")
        print(e)

    # Load all submissions from conference
    # all_reviews = client.get_all_submission_reviews(venue="ICLR.cc", year=2024, type="Conference")
    # review_sentences = [sent for review in all_reviews for sent in review.sentences]
    # all_reviews_from_ICLR2024 = pd.DataFrame(review_sentences, columns=["sentences"])
    # all_reviews_from_ICLR2024.to_csv("data/ICLR2024_all_reviews.csv", index=False, encoding="utf-8")

    # "zzv4Bf50RW", "KS8mIvetg2", "aVh9KRZdRk", "gojL67CfS8"
    ids = ["zzv4Bf50RW", "KS8mIvetg2", "aVh9KRZdRk", "gojL67CfS8"]

    # The sentences we get are the same. However, with create_testset() we don't get any metadata,
    # With get_reviews() we get a list of Review Object which contain the variable sentences where all the sentences of
    # the review are stored
    # To get all the sentences from the list of reviews we could also use something like:
    # [print(sent) for review in reviews for sent in review.sentences]
    test_set = client.create_testset(ids=ids)
    # reviews = client.get_reviews(id=ids)

    test_set = pd.DataFrame(test_set, columns=["sentences"])
    test_set.to_csv("data/test_set_2.csv", index=False, encoding="utf-8")
