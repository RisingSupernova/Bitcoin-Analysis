#!/usr/bin/env python3
"""Post Bitcoin Daily Allocation Report to Twitter/X with screenshots.

Usage:
    python twitter_poster.py              # Post to Twitter (requires env vars)
    python twitter_poster.py --dry-run    # Generate screenshots only, skip posting
    python twitter_poster.py --force      # Bypass idempotency guard
"""

import argparse
import glob
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
HEADER_TEMPLATE = Path(__file__).parent / "header_template.html"

# Environment variables
TWITTER_API_KEY = os.environ.get("TWITTER_API_KEY", "")
TWITTER_API_SECRET = os.environ.get("TWITTER_API_SECRET", "")
TWITTER_ACCESS_TOKEN = os.environ.get("TWITTER_ACCESS_TOKEN", "")
TWITTER_ACCESS_SECRET = os.environ.get("TWITTER_ACCESS_SECRET", "")
IMGBB_API_KEY = os.environ.get("IMGBB_API_KEY", "")


def find_latest_file(pattern):
    """Find the most recently modified file matching a glob pattern."""
    files = sorted(glob.glob(str(DATA_DIR / pattern)), key=os.path.getmtime, reverse=True)
    return Path(files[0]) if files else None


def load_metadata():
    """Load the latest twitter metadata JSON."""
    meta_file = find_latest_file("twitter_meta_*.json")
    if not meta_file:
        print("No twitter_meta_*.json found in data/")
        sys.exit(1)
    print(f"  Loading metadata: {meta_file.name}")
    return json.loads(meta_file.read_text()), meta_file


def find_report_html(date_str):
    """Find the report HTML matching the metadata date, or latest."""
    report_file = DATA_DIR / f"report_{date_str}.html"
    if report_file.exists():
        return report_file
    # Fallback: latest report
    fallback = find_latest_file("report_*.html")
    if fallback:
        print(f"  Warning: report_{date_str}.html not found, using {fallback.name}")
        return fallback
    print("No report HTML found in data/")
    sys.exit(1)


def regime_css_class(regime):
    """Map market regime to CSS class."""
    r = regime.upper()
    if "BEAR" in r:
        return "badge-bear"
    if "BULL" in r:
        return "badge-bull"
    if "ACCUM" in r:
        return "badge-accumulation"
    return "badge-unknown"


def cycle_css_class(cycle):
    """Map cycle phase to CSS class."""
    c = cycle.upper().replace(" ", "_")
    mapping = {
        "EARLY_BEAR": "badge-early-bear",
        "LATE_BEAR": "badge-late-bear",
        "EARLY_BULL": "badge-early-bull",
        "MID_BULL": "badge-mid-bull",
        "LATE_BULL": "badge-late-bull",
    }
    return mapping.get(c, "badge-unknown")


def format_date(date_str):
    """Convert YYYYMMDD to readable date like 'Tuesday, March 10, 2026'."""
    try:
        dt = datetime.strptime(date_str, "%Y%m%d")
        return dt.strftime("%A, %B %d, %Y").replace(" 0", " ")
    except ValueError:
        return date_str


def take_header_screenshot(metadata):
    """Render header template with metadata and return screenshot bytes."""
    from playwright.sync_api import sync_playwright

    template = HEADER_TEMPLATE.read_text()
    html = template.replace("{{DATE}}", format_date(metadata["date"]))
    html = html.replace("{{REGIME}}", metadata["market_regime"])
    html = html.replace("{{CYCLE_PHASE}}", metadata["cycle_phase"].replace("_", " "))
    html = html.replace("{{REGIME_CLASS}}", regime_css_class(metadata["market_regime"]))
    html = html.replace("{{CYCLE_CLASS}}", cycle_css_class(metadata["cycle_phase"]))

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        f.write(html)
        tmp_path = f.name

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(
                viewport={"width": 600, "height": 315},
                device_scale_factor=2,
            )
            page.goto(f"file://{tmp_path}")
            page.wait_for_load_state("networkidle")
            screenshot = page.screenshot(type="png")
            browser.close()
        return screenshot
    finally:
        os.unlink(tmp_path)


def take_report_screenshot(report_path):
    """Take a full-page screenshot of the report HTML. Returns (bytes, format)."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": 800, "height": 600},
            device_scale_factor=2,
        )
        page.goto(f"file://{report_path}")
        page.wait_for_load_state("networkidle")

        # Take PNG screenshot first
        screenshot = page.screenshot(type="png", full_page=True)
        fmt = "png"

        # If over 10MB, re-render as JPEG
        if len(screenshot) > 10 * 1024 * 1024:
            print(f"  PNG is {len(screenshot) / 1024 / 1024:.1f}MB, converting to JPEG")
            screenshot = page.screenshot(type="jpeg", quality=90, full_page=True)
            fmt = "jpeg"

        browser.close()
    return screenshot, fmt


def upload_to_imgbb(image_bytes, fmt="png"):
    """Upload image to imgBB. Returns direct image URL."""
    import requests

    if not IMGBB_API_KEY:
        print("IMGBB_API_KEY not set")
        sys.exit(1)

    import base64
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = requests.post(
        "https://api.imgbb.com/1/upload",
        data={"key": IMGBB_API_KEY, "image": b64},
        timeout=60,
    )

    if response.status_code != 200:
        print(f"imgBB upload failed: {response.status_code} {response.text[:200]}")
        sys.exit(1)

    data = response.json()
    link = data["data"]["url"]
    print(f"  imgBB upload OK: {link}")
    return link


def post_tweets(subject, header_img, imgur_url, dry_run=False):
    """Post 2-tweet thread: header + report link."""
    tweet1_text = f"{subject}\n\nLink to full report below."
    tweet2_text = f"Full report:\n{imgur_url}"

    print(f"\n  Tweet 1 ({len(tweet1_text)} chars):\n    {tweet1_text.replace(chr(10), chr(10) + '    ')}")
    print(f"\n  Tweet 2 ({len(tweet2_text)} chars):\n    {tweet2_text.replace(chr(10), chr(10) + '    ')}")

    if dry_run:
        print("\n  [DRY RUN] Skipping Twitter posting")
        return None, None

    import tweepy

    # v1.1 API for media upload
    auth = tweepy.OAuth1UserHandler(
        TWITTER_API_KEY, TWITTER_API_SECRET,
        TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET,
    )
    api = tweepy.API(auth)

    # v2 Client for tweet creation
    client = tweepy.Client(
        consumer_key=TWITTER_API_KEY,
        consumer_secret=TWITTER_API_SECRET,
        access_token=TWITTER_ACCESS_TOKEN,
        access_token_secret=TWITTER_ACCESS_SECRET,
    )

    # Upload header image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(header_img)
        tmp_media = f.name

    try:
        media = api.media_upload(filename=tmp_media)
        print(f"  Media uploaded: {media.media_id}")
    finally:
        os.unlink(tmp_media)

    # Tweet 1: header + subject
    response1 = client.create_tweet(text=tweet1_text, media_ids=[media.media_id])
    tweet1_id = response1.data["id"]
    print(f"  Tweet 1 posted: {tweet1_id}")

    # Brief pause for thread reliability
    time.sleep(2)

    # Tweet 2: reply with report link
    try:
        response2 = client.create_tweet(
            text=tweet2_text,
            in_reply_to_tweet_id=tweet1_id,
        )
        tweet2_id = response2.data["id"]
        print(f"  Tweet 2 posted: {tweet2_id}")
    except Exception as e:
        print(f"  Warning: Tweet 2 failed: {e}")
        tweet2_id = None

    return tweet1_id, tweet2_id


def log_to_summary(message):
    """Write a message to GitHub Actions step summary if available."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(f"\n### Twitter Poster\n{message}\n")


def main():
    parser = argparse.ArgumentParser(description="Post BTC report to Twitter")
    parser.add_argument("--dry-run", action="store_true", help="Generate screenshots only")
    parser.add_argument("--force", action="store_true", help="Bypass idempotency guard")
    args = parser.parse_args()

    print("=" * 50)
    print("Twitter Poster for Bitcoin Daily Report")
    print("=" * 50)

    # 1. Load metadata
    metadata, meta_file = load_metadata()
    date_str = metadata["date"]

    # 2. Idempotency guard
    posted_file = DATA_DIR / f"twitter_posted_{date_str}.json"
    if posted_file.exists() and not args.force:
        print(f"  Already posted today ({posted_file.name}). Use --force to re-post.")
        return

    # 3. Validate credentials (skip for dry-run)
    if not args.dry_run:
        missing = []
        if not TWITTER_API_KEY:
            missing.append("TWITTER_API_KEY")
        if not TWITTER_API_SECRET:
            missing.append("TWITTER_API_SECRET")
        if not TWITTER_ACCESS_TOKEN:
            missing.append("TWITTER_ACCESS_TOKEN")
        if not TWITTER_ACCESS_SECRET:
            missing.append("TWITTER_ACCESS_SECRET")
        if not IMGBB_API_KEY:
            missing.append("IMGBB_API_KEY")
        if missing:
            print(f"  Missing env vars: {', '.join(missing)}")
            log_to_summary(f"Skipped: missing credentials ({', '.join(missing)})")
            sys.exit(1)

    # 4. Generate header screenshot
    print("\n📸 Generating header screenshot...")
    header_img = take_header_screenshot(metadata)
    print(f"  Header: {len(header_img) / 1024:.0f}KB PNG")

    # 5. Generate full report screenshot
    print("\n📸 Generating report screenshot...")
    report_path = find_report_html(date_str)
    report_img, report_fmt = take_report_screenshot(report_path)
    print(f"  Report: {len(report_img) / 1024:.0f}KB {report_fmt.upper()}")

    # Save screenshots in dry-run mode
    if args.dry_run:
        header_out = DATA_DIR / f"header_{date_str}.png"
        header_out.write_bytes(header_img)
        report_out = DATA_DIR / f"full_report_{date_str}.{report_fmt}"
        report_out.write_bytes(report_img)
        print(f"\n  Saved: {header_out.name}, {report_out.name}")

    # 6. Upload full report to Imgur
    print("\n📤 Uploading report to imgBB...")
    if args.dry_run:
        imgur_url = "https://i.ibb.co/DRYRUN/report.png"
        print("  [DRY RUN] Skipping imgBB upload")
    else:
        imgur_url = upload_to_imgbb(report_img, report_fmt)

    # 7. Post tweets
    print("\n🐦 Posting to Twitter...")
    tweet1_id, tweet2_id = post_tweets(
        metadata["email_subject"],
        header_img,
        imgur_url,
        dry_run=args.dry_run,
    )

    # 8. Save posted marker
    if not args.dry_run and tweet1_id:
        posted_data = {
            "date": date_str,
            "tweet1_id": str(tweet1_id),
            "tweet2_id": str(tweet2_id) if tweet2_id else None,
            "imgur_url": imgur_url,
            "posted_at": datetime.now(timezone.utc).isoformat(),
        }
        posted_file.write_text(json.dumps(posted_data, indent=2))
        print(f"\n  Guard file saved: {posted_file.name}")
        log_to_summary(f"Posted successfully. [Tweet](https://x.com/twocommapauper/status/{tweet1_id})")

    print("\n✅ Done!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()
        log_to_summary(f"Failed: {e}")
        sys.exit(1)
