import json
import os
import sys
import tempfile
from io import BytesIO
from pathlib import Path
import requests
import openreview
import pymupdf4llm
import openai
from dotenv import load_dotenv
import time
from tqdm import tqdm

# Load environment variables from every location the project uses, with
# explicit paths. A bare load_dotenv() cannot be relied on here: it discovers
# .env by walking up from the *calling frame's* file, so under "python -m" it
# finds indiaml/indiaml/.env (OpenReview credentials) and stops, never reaching
# indiaml/.env (OPENROUTER_API_KEY, per indiaml/.env.sample). Resolving both
# explicitly makes this independent of how the script is invoked and of the CWD.
_PKG_DIR = Path(__file__).resolve().parents[1]      # indiaml/indiaml
_PROJECT_DIR = Path(__file__).resolve().parents[2]  # indiaml
for _env_path in (_PKG_DIR / ".env", _PROJECT_DIR / ".env", Path.cwd() / ".env"):
    if _env_path.is_file():
        load_dotenv(_env_path)

PLACEHOLDER_API_KEY = "sk-or-v1-..."

# Model used for summarization. The previous default, google/gemini-flash-1.5-8b,
# has been retired and now returns HTTP 404 "No endpoints found", so it is
# overridable without editing the source.
SUMMARY_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash-lite")


def get_openrouter_key():
    """Return the OpenRouter API key, or None if it is missing or a placeholder."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key or key == PLACEHOLDER_API_KEY:
        return None
    return key


# Set up OpenAI client with OpenRouter. Built lazily so that importing this
# module (or reusing its download helpers) does not require an API key.
_client = None


def get_client():
    global _client
    if _client is None:
        _client = openai.OpenAI(
            api_key=get_openrouter_key() or PLACEHOLDER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
    return _client


# Function to download PDF
_openreview_client = None


def get_openreview_client():
    """Return a cached, authenticated OpenReview client.

    OpenReview returns HTTP 403 ("Access to this page is restricted") for
    anonymous PDF requests, so downloading requires a logged-in client.
    Returns None if credentials are absent or login fails, in which case the
    caller falls back to an unauthenticated request.
    """
    global _openreview_client
    if _openreview_client is not None:
        return _openreview_client or None

    username = os.environ.get("OPENREVIEW_USERNAME")
    password = os.environ.get("OPENREVIEW_PASSWORD")
    if not username or not password:
        print(
            "Warning: OPENREVIEW_USERNAME/OPENREVIEW_PASSWORD are not set. "
            "OpenReview denies anonymous PDF downloads, so summaries will be skipped."
        )
        _openreview_client = False
        return None

    try:
        _openreview_client = openreview.api.OpenReviewClient(
            baseurl="https://api2.openreview.net",
            username=username,
            password=password,
        )
    except Exception as e:
        print(f"Error authenticating to OpenReview: {e}")
        _openreview_client = False
        return None

    return _openreview_client


class RateLimited(Exception):
    """OpenReview refused the download because every route was rate limited."""


def download_pdf(pdf_url, paper_id=None):
    """Download the PDF for a paper, authenticating to OpenReview.

    Tries each known download route in turn, all via plain requests:

      1. https://openreview.net/pdf?id=<paper_id>            (26 per hour)
      2. https://api2.openreview.net/attachment?id=..&name=pdf (36 per hour)
      3. the stored pdf_url

    OpenReview meters these routes separately, so using both roughly doubles the
    hourly throughput. Route 1 is preferred over the stored pdf_url because most
    stored values use a content-hash form (/pdf/<sha1>.pdf) that now 404s.

    Everything goes through requests rather than openreview-py's get_attachment:
    that helper retries internally before surfacing a 429, which blocks for tens
    of minutes with no output.

    Raises RateLimited only when every route reported 429, so the caller can
    stop cleanly and keep the work already done.
    """
    client = get_openreview_client()

    headers = {}
    if client is not None and getattr(client, "token", None):
        headers["Authorization"] = f"Bearer {client.token.replace('Bearer ', '')}"

    candidates = []
    if paper_id:
        candidates.append(f"https://openreview.net/pdf?id={paper_id}")
        candidates.append(
            f"https://api2.openreview.net/attachment?id={paper_id}&name=pdf"
        )
    if pdf_url and pdf_url not in candidates:
        candidates.append(pdf_url)

    saw_rate_limit = False
    for url in candidates:
        try:
            response = requests.get(url, headers=headers, timeout=90)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading PDF from {url}: {e}")
            continue
        if response.status_code == 429:
            saw_rate_limit = True
            retry_after = response.headers.get("retry-after", "?")
            print(f"Rate limited on {url} (retry after {retry_after}s).")
            continue
        if response.status_code == 404:
            print(f"PDF not found at {url}.")
            continue
        try:
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error downloading PDF from {url}: {e}")
            continue
        return BytesIO(response.content)

    if saw_rate_limit:
        raise RateLimited(f"all download routes rate limited for {paper_id}")

    return None


# Function to convert PDF to Markdown using pymupdf4llm
def convert_pdf_to_markdown(pdf_stream, num_pages=3):
    """Convert the first `num_pages` of the PDF to Markdown using pymupdf4llm."""
    if pdf_stream is None:
        return ""
    
    # Create a temporary file to save the PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_stream.read())
        temp_pdf_path = temp_pdf.name
    
    try:
        # Call the to_markdown function with the temporary file path
        markdown = pymupdf4llm.to_markdown(temp_pdf_path, pages=range(0, num_pages))
        return markdown
    except Exception as e:
        print(f"Error converting PDF to markdown: {e}")
        return ""
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

def summarize_paper_goal(text):
    """Summarize the paper's goal. Returns None if no summary could be produced.

    Returning None rather than a sentinel string matters: the caller writes this
    value straight into paper_content and saves the file, so returning an error
    message here would persist "Error generating summary." as though it were a
    real summary and mark the paper as done.
    """
    if not text.strip():
        print("No text available for summarization.")
        return None

    try:
        response = get_client().chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[
                {"role": "system", "content": "You are extremely efficient and formal at writing summaries from papers. You try to write summaries intended to be read by an audience on a webpage."},
                {"role": "user", "content": f"Based on the following excerpt from a research paper, summarize the paper's goal, keep it very brief, within 2 to 3 sentences:\n\n{text[:4000]}"}  # Limit text length to avoid token limits
            ]
        )
    except Exception as e:
        print(f"Error calling the API for summarization: {e}")
        return None

    summary = (response.choices[0].message.content or "").strip()
    if not summary:
        print("Model returned an empty summary.")
        return None
    return summary

# Values that earlier versions of this script wrote into paper_content when a
# step failed. They are truthy, so a naive "already has content" check treats
# them as real summaries and skips those papers permanently.
FAILURE_SENTINELS = frozenset({
    "Error generating summary.",
    "No text available for summarization.",
})


def needs_summary(paper):
    """True if the paper has no usable summary yet."""
    content = (paper.get("paper_content") or "").strip()
    return not content or content in FAILURE_SENTINELS


def save_papers(file_path, papers):
    """Write papers back to disk atomically, so an interrupted run cannot
    truncate an existing venue file."""
    tmp_path = f"{file_path}.tmp"
    try:
        with open(tmp_path, "w") as file:
            json.dump(papers, file, indent=2)
        os.replace(tmp_path, file_path)
        return True
    except Exception as e:
        print(f"Error writing to file {file_path}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


def process_venue_file(file_path, tracker_dir):
    """Process a venue JSON file, summarizing papers that lack a usable summary.

    Returns (updated_count, rate_limited).
    """
    print(f"\nProcessing file: {file_path}")
    
    # Read the venue JSON file
    try:
        with open(file_path, "r") as file:
            papers = json.load(file)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return 0, False
    
    # Counter for papers that were updated
    updated_count = 0
    rate_limited = False
    
    # Process each paper
    for paper in tqdm(papers, desc="Papers"):
        # Skip if the paper already has a real summary. Failure sentinels written
        # by earlier runs count as missing, not as content, otherwise they are
        # truthy and would be skipped forever.
        if not needs_summary(paper):
            continue
        
        print(f"\nProcessing paper: {paper['paper_title']}")
        pdf_url = paper["pdf_url"]
        
        # Step 1: Download the PDF
        try:
            pdf_stream = download_pdf(pdf_url, paper.get("paper_id"))
        except RateLimited as e:
            print(f"\nOpenReview rate limit reached: {e}")
            rate_limited = True
            break
        if pdf_stream is None:
            print(f"Skipping paper due to download failure: {paper['paper_title']}")
            continue
        
        # Step 2: Extract text from the first 3 pages
        extracted_text = convert_pdf_to_markdown(pdf_stream, num_pages=3)
        if not extracted_text.strip():
            print("No text extracted from the first 3 pages.")
            continue
        
        # Step 3: Summarize the paper's goal using the API
        summary = summarize_paper_goal(extracted_text)
        if not summary:
            print(f"Skipping paper due to summarization failure: {paper['paper_title']}")
            continue
        print(f"Summary generated: {summary}")
        
        # Step 4: Update the paper with the summary, then persist immediately.
        # Saving per paper rather than per file means a rate limit or crash
        # cannot discard summaries that were already paid for.
        paper["paper_content"] = summary
        updated_count += 1
        save_papers(file_path, papers)
        
        # Add a small delay to avoid rate limiting
        time.sleep(1)
    
    if updated_count > 0:
        print(f"Updated {updated_count} papers in {file_path}")
    else:
        print(f"No papers needed updating in {file_path}")
    
    return updated_count, rate_limited

def main():
    # Fail fast rather than downloading every PDF and then failing on each
    # summarization call with an unusable key.
    if get_openrouter_key() is None:
        print(
            "OPENROUTER_API_KEY is not set (or is still the placeholder value).\n"
            "Set it in indiaml/indiaml/.env or export it, then re-run."
        )
        return 1

    if get_openreview_client() is None:
        print(
            "Cannot authenticate to OpenReview. OpenReview refuses anonymous PDF\n"
            "downloads (HTTP 403), so no summaries could be generated. Set\n"
            "OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD in indiaml/indiaml/.env."
        )
        return 1

    # Set tracker directory from CLI args or use default
    tracker_dir = sys.argv[1] if len(sys.argv) > 1 else "../ui/indiaml-tracker/public/tracker"
    
    # Ensure the tracker directory exists
    if not os.path.exists(tracker_dir):
        print(f"Tracker directory not found: {tracker_dir}")
        return 1
    
    # Read the index.json file
    index_path = os.path.join(tracker_dir, "index.json")
    if not os.path.exists(index_path):
        print(f"Index file not found: {index_path}")
        return 1
    
    try:
        with open(index_path, "r") as index_file:
            index_data = json.load(index_file)
    except Exception as e:
        print(f"Error reading index file: {e}")
        return 1
    
    print(f"Found {len(index_data)} venue-year entries in the index")
    
    # Process each venue-year file
    total_updated = 0
    hit_rate_limit = False
    for entry in tqdm(index_data, desc="Venues"):
        file_name = entry["file"]
        file_path = os.path.join(tracker_dir, file_name)
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        # Process the venue file
        updated, rate_limited = process_venue_file(file_path, tracker_dir)
        total_updated += updated
        if rate_limited:
            hit_rate_limit = True
            break
    
    print(f"\nSummary generation complete. Added summaries to {total_updated} papers across all venues.")

    if hit_rate_limit:
        remaining = count_remaining(tracker_dir, index_data)
        print(
            f"\nStopped early because OpenReview's download rate limit was reached.\n"
            f"{remaining} papers still need a summary. Everything generated so far has\n"
            f"been saved. Re-run this command after the limit resets (roughly an hour)\n"
            f"and it will resume where it left off."
        )
        return 2
    return 0


def count_remaining(tracker_dir, index_data):
    """Count papers across all venue files that still need a summary."""
    remaining = 0
    for entry in index_data:
        path = os.path.join(tracker_dir, entry["file"])
        if not os.path.exists(path):
            continue
        try:
            with open(path) as fh:
                remaining += sum(1 for p in json.load(fh) if needs_summary(p))
        except Exception:
            pass
    return remaining

if __name__ == "__main__":
    sys.exit(main() or 0)