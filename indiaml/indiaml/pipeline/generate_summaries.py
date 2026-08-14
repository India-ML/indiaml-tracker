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

# Load environment variables. Load the package-local .env as well as any .env in
# the current working directory, so the script works regardless of where it is
# invoked from (matches base_adapter.py and db_config.py).
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
load_dotenv()

PLACEHOLDER_API_KEY = "sk-or-v1-..."


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


def download_pdf(pdf_url, paper_id=None):
    """Download the PDF for a paper, authenticating to OpenReview when possible."""
    client = get_openreview_client()

    # Preferred path: ask OpenReview for the attachment directly.
    if client is not None and paper_id:
        try:
            return BytesIO(client.get_attachment(id=paper_id, field_name="pdf"))
        except Exception as e:
            print(f"Could not fetch attachment for {paper_id} ({e}); falling back to URL.")

    headers = {}
    if client is not None and getattr(client, "token", None):
        headers["Authorization"] = f"Bearer {client.token.replace('Bearer ', '')}"

    try:
        response = requests.get(pdf_url, headers=headers, timeout=60)
        response.raise_for_status()
        return BytesIO(response.content)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF from {pdf_url}: {e}")
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
    """Uses the OpenAI Chat API to summarize the paper's goal based on the extracted text."""
    if not text.strip():
        return "No text available for summarization."
    
    try:
        response = get_client().chat.completions.create(
            model="google/gemini-flash-1.5-8b",  # Change this to your desired model if needed
            messages=[
                {"role": "system", "content": "You are extremely efficient and formal at writing summaries from papers. You try to write summaries intended to be read by an audience on a webpage."},
                {"role": "user", "content": f"Based on the following excerpt from a research paper, summarize the paper's goal, keep it very brief, within 2 to 3 sentences:\n\n{text[:4000]}"}  # Limit text length to avoid token limits
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling the API for summarization: {e}")
        return "Error generating summary."

def process_venue_file(file_path, tracker_dir):
    """Process a single venue JSON file and add summaries to papers that don't have them."""
    print(f"\nProcessing file: {file_path}")
    
    # Read the venue JSON file
    try:
        with open(file_path, "r") as file:
            papers = json.load(file)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return 0
    
    # Counter for papers that were updated
    updated_count = 0
    
    # Process each paper
    for paper in tqdm(papers, desc="Papers"):
        # Skip if the paper already has content
        if "paper_content" in paper and paper["paper_content"]:
            continue
        
        print(f"\nProcessing paper: {paper['paper_title']}")
        pdf_url = paper["pdf_url"]
        
        # Step 1: Download the PDF
        pdf_stream = download_pdf(pdf_url, paper.get("paper_id"))
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
        print(f"Summary generated: {summary}")
        
        # Step 4: Update the paper with the summary
        paper["paper_content"] = summary
        updated_count += 1
        
        # Add a small delay to avoid rate limiting
        time.sleep(1)
    
    # Save the updated papers back to the file
    if updated_count > 0:
        try:
            with open(file_path, "w") as file:
                json.dump(papers, file, indent=2)
            print(f"Updated {updated_count} papers in {file_path}")
        except Exception as e:
            print(f"Error writing to file {file_path}: {e}")
    else:
        print(f"No papers needed updating in {file_path}")
    
    return updated_count

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
    for entry in tqdm(index_data, desc="Venues"):
        file_name = entry["file"]
        file_path = os.path.join(tracker_dir, file_name)
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        # Process the venue file
        updated = process_venue_file(file_path, tracker_dir)
        total_updated += updated
    
    print(f"\nSummary generation complete. Added summaries to {total_updated} papers across all venues.")
    return 0

if __name__ == "__main__":
    sys.exit(main() or 0)