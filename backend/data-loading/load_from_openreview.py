import argparse
import pandas as pd

from loaders.openreview_loader import OpenReviewLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("username", help="Openreview username")
    parser.add_argument("password", help="Openreview password")

    args = parser.parse_args()

    client = OpenReviewLoader(username=args.username, password=args.password)

    ids = ["zzv4Bf50RW", "KS8mIvetg2", "aVh9KRZdRk", "gojL67CfS8"]

    # The sentences we get are the same. However, with create_testset() we don't get any metadata,
    # With get_reviews() we get a list of Review Object which contain the variable sentences where all the sentences of
    # the review are stored
    # To get all the sentences from the list of reviews we could also use something like:
    # [print(sent) for review in reviews for sent in review.sentences]
    test_set = client.create_testset(ids=ids)
    reviews = client.get_reviews(id=["zzv4Bf50RW", "KS8mIvetg2", "aVh9KRZdRk", "gojL67CfS8"])

    test_set = pd.DataFrame(test_set, columns=["sentences"])
    test_set.to_csv("data/test_set.csv", index=False, encoding="utf-8")
