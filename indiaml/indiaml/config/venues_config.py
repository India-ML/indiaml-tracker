from pydantic import BaseModel
from typing import List

class VenueConfig(BaseModel):
    conference: str
    year: int
    track: str
    source_adapter: str  # e.g., "openreview"
    source_id: str       # e.g., "NeurIPS.cc/2024/Conference"
    adapter_class: str   # e.g., "NeurIPSAdapter", "ICMLAdapter", "ICAIAdapter"

# Define your venue configurations here.
#
# Only the venue currently being processed should be active: process_venue.py
# iterates every entry below, and each venue is built into its own database
# (see data/venues-<conference>-<year>-v*.db), selected via INDIAML_DATABASE_URL.
# Completed venues are kept commented out as a record of what has been run.
VENUE_CONFIGS: List[VenueConfig] = [
    VenueConfig(
        conference="ICML",
        year=2026,
        track="Conference",
        source_adapter="openreview",
        source_id="ICML.cc/2026/Conference",
        adapter_class="ICAIAdapter"
    ),
    # VenueConfig(
    #     conference="ICLR",
    #     year=2026,
    #     track="Conference",
    #     source_adapter="openreview",
    #     source_id="ICLR.cc/2026/Conference",
    #     adapter_class="ICAIAdapter"
    # ),
    # NeurIPS 2026 has no accepted papers yet; the conference runs Dec 06 2026.
    # VenueConfig(
    #     conference="NeurIPS",
    #     year=2025,
    #     track="Conference",
    #     source_adapter="openreview",
    #     source_id="NeurIPS.cc/2025/Conference",
    #     adapter_class="NeurIPSAdapter"
    # ),
    # VenueConfig(
    #     conference="ICML",
    #     year=2025,
    #     track="Conference",
    #     source_adapter="openreview",
    #     source_id="ICML.cc/2025/Conference",
    #     adapter_class="ICAIAdapter"
    # ),
    # VenueConfig(
    #     conference="ICLR",
    #     year=2025,
    #     track="Conference",
    #     source_adapter="openreview",
    #     source_id="ICLR.cc/2025/Conference",
    #     adapter_class="ICAIAdapter"
    # )
]
