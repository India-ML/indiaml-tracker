from sqlalchemy.orm import Session, joinedload
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple
from ..venue.venudao import VenueDB
from ..models.models import PaperAuthor, Paper, Author
from ..models.dto import AuthorDTO
from ..config.venues_config import VENUE_CONFIGS
from ..venue_adapters.adapter_factory import get_adapter
from ..config.db_config import init_db
import logging
import sys

# SETUP LOGGING
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

# Profiles are fetched in bulk; openreview batches these into 1000-id requests.
PROFILE_BATCH_SIZE = 1000


def _venue_key(paper: Paper) -> Tuple[str, int, str]:
    return (paper.venue_info.conference, paper.venue_info.year, paper.venue_info.track)


def _paper_author_ids(paper: Paper) -> List[str]:
    """Author openreview ids for a paper, preserving order."""
    return [
        author['openreview_id']
        for author in (paper.raw_authors or [])
        if 'openreview_id' in author
    ]


def _chunks(items: List[str], size: int) -> Iterable[List[str]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _prefetch_authors(adapter, papers: List[Paper]) -> Dict[str, AuthorDTO]:
    """
    Fetch every author profile for a venue up front.

    The per-paper lookups this feeds are identical to fetching per paper, but it
    avoids issuing one profile request per paper.
    """
    unique_ids: List[str] = []
    seen = set()
    for paper in papers:
        for author_id in _paper_author_ids(paper):
            if author_id not in seen:
                seen.add(author_id)
                unique_ids.append(author_id)

    logger.info(f"Prefetching {len(unique_ids)} unique author profiles.")

    profiles: Dict[str, AuthorDTO] = {}
    for batch in _chunks(unique_ids, PROFILE_BATCH_SIZE):
        for author_dto in adapter.fetch_authors(batch):
            if author_dto.openreview_id:
                profiles[author_dto.openreview_id] = author_dto
        logger.info(f"Resolved {len(profiles)}/{len(unique_ids)} author profiles.")

    return profiles


def process_authors():
    """Process authors from stored papers and save them to the database."""
    try:
        with VenueDB() as db:
            session: Session = db.session
            # Fetch all papers with related venue_info using joinedload to optimize queries
            papers: List[Paper] = session.query(Paper).options(
                joinedload(Paper.venue_info)
            ).all()

            logger.info(f"Processing authors for {len(papers)} papers.")

            # Group papers per venue so profiles can be fetched once per venue.
            papers_by_venue: Dict[Tuple[str, int, str], List[Paper]] = {}
            for paper in papers:
                if not paper.venue_info:
                    logger.warning(f"Missing venue information for paper ID: {paper.id}")
                    continue
                papers_by_venue.setdefault(_venue_key(paper), []).append(paper)

            for (conference, year, track), venue_papers in papers_by_venue.items():
                # Retrieve the corresponding VenueConfig based on conference, year, and track
                venue_config = next(
                    (cfg for cfg in VENUE_CONFIGS
                     if cfg.conference == conference and
                        cfg.year == year and
                        cfg.track == track),
                    None
                )

                if not venue_config:
                    logger.warning(f"No VenueConfig found for conference '{conference}', year '{year}', and track '{track}'. Skipping {len(venue_papers)} papers.")
                    continue

                logger.info(f"Processing {len(venue_papers)} papers for {conference} {year} {track}.")

                # Get the appropriate adapter once per venue instead of once per paper.
                adapter = get_adapter(venue_config)
                author_profiles = _prefetch_authors(adapter, venue_papers)

                for processed, paper in enumerate(venue_papers, start=1):
                    logger.debug(f"Processing paper ID: {paper.id}")

                    # Use the prefetched profiles for this paper's authors
                    detailed_authors = [
                        author_profiles[author_id]
                        for author_id in _paper_author_ids(paper)
                        if author_id in author_profiles
                    ]

                    if not detailed_authors:
                        logger.warning(f"No detailed authors fetched for paper ID: {paper.id}")
                        # Fallback to raw authors if detailed authors are not available
                        detailed_authors = [
                            AuthorDTO(
                                name=author.get('name'),
                                email=author.get('email'),
                                openreview_id=author.get('openreview_id'),
                                orcid=author.get('orcid'),
                                google_scholar_link=author.get('google_scholar_link'),
                                linkedin=author.get('linkedin'),
                                homepage=author.get('homepage'),
                                history=author.get('history')  # Ensure history is included if available
                            )
                            for author in (paper.raw_authors or [])
                        ]

                    if not detailed_authors:
                        logger.warning(f"No authors to process for paper ID: {paper.id}")
                        continue

                    for idx, author_dto in enumerate(detailed_authors):
                        try:
                            author = db.get_or_create_author(author_dto)

                            # Create or update PaperAuthor association with sequence position
                            paper_author = session.query(PaperAuthor).filter_by(
                                paper_id=paper.id,
                                author_id=author.id
                            ).one_or_none()

                            if not paper_author:
                                paper_author = PaperAuthor(
                                    paper=paper,
                                    author=author,
                                    position=idx  # Sequence starts at 0
                                )
                                session.add(paper_author)
                            else:
                                paper_author.position = idx  # Update position if necessary

                        except Exception as e:
                            session.rollback()
                            logger.error(f"Error processing author '{author_dto.name}' for paper {paper.id}: {e}")

                    session.commit()

                    if processed % 500 == 0:
                        logger.info(f"Processed {processed}/{len(venue_papers)} papers for {conference} {year} {track}.")

            logger.info("All authors processed successfully.")

    except Exception as e:
        logger.error(f"Error in processing authors: {e}")
        raise

if __name__ == "__main__":
    init_db()  # Ensure the database is initialized
    process_authors()
