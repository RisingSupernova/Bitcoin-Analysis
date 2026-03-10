#!/usr/bin/env python3
"""Retweet the daily Bitcoin report for additional reach.

Reads the twitter_posted_{date}.json guard file to find the tweet ID,
then retweets it. Designed to run 12 hours after the original post.
"""

import glob
import json
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def main():
    print("🔁 Retweet Daily Report")

    # Find latest posted file
    files = sorted(glob.glob(str(DATA_DIR / "twitter_posted_*.json")), key=os.path.getmtime, reverse=True)
    if not files:
        print("  No twitter_posted_*.json found. Nothing to retweet.")
        return

    posted = json.loads(Path(files[0]).read_text())
    tweet_id = posted.get("tweet1_id")
    if not tweet_id:
        print("  No tweet1_id in posted file. Skipping.")
        return

    if posted.get("retweeted"):
        print(f"  Already retweeted ({posted['date']}). Skipping.")
        return

    # Check credentials
    api_key = os.environ.get("TWITTER_API_KEY", "")
    api_secret = os.environ.get("TWITTER_API_SECRET", "")
    access_token = os.environ.get("TWITTER_ACCESS_TOKEN", "")
    access_secret = os.environ.get("TWITTER_ACCESS_SECRET", "")

    if not all([api_key, api_secret, access_token, access_secret]):
        print("  Missing Twitter credentials. Skipping.")
        sys.exit(1)

    import tweepy

    client = tweepy.Client(
        consumer_key=api_key,
        consumer_secret=api_secret,
        access_token=access_token,
        access_token_secret=access_secret,
    )

    client.retweet(tweet_id)
    print(f"  Retweeted: {tweet_id}")

    # Mark as retweeted
    posted["retweeted"] = True
    Path(files[0]).write_text(json.dumps(posted, indent=2))
    print("  Guard file updated.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"  Error: {e}")
        sys.exit(1)
