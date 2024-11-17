import argparse
import pandas as pd

from loaders.openreview_loader import OpenReviewLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", help="Openreview username")
    parser.add_argument("-pw", help="Openreview password")

    args = parser.parse_args()

    client = OpenReviewLoader(username=args.username, password=args.password)
    test_set = client.create_testset(ids=["zzv4Bf50RW", "KS8mIvetg2"])

    test_set = pd.DataFrame(test_set, columns=["sentences"])
    test_set.to_csv("data/test_set.csv", index=False, encoding="utf-8")
