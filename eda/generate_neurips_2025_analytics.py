#!/usr/bin/env python3
"""
Generate NeurIPS 2025 analytics without accept_type dependency
"""
import sqlite3
import json
from collections import defaultdict

def generate_analytics(db_path, output_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get conference totals
    cursor.execute("""
        SELECT COUNT(DISTINCT p.id) as total_papers, 
               COUNT(DISTINCT pa.author_id) as total_authors
        FROM papers p
        JOIN venue_infos v ON p.venue_info_id = v.id
        JOIN paper_authors pa ON p.id = pa.paper_id
        WHERE v.conference = 'NeurIPS' 
        AND v.year = 2025 
        AND v.track = 'Conference'
        AND p.status = 'accepted'
    """)
    total_papers, total_authors = cursor.fetchone()
    
    # Get country statistics
    cursor.execute("""
        SELECT 
            COALESCE(pa.affiliation_country, 'UNK') as country,
            COUNT(DISTINCT p.id) as paper_count,
            COUNT(DISTINCT pa.author_id) as author_count
        FROM papers p
        JOIN venue_infos v ON p.venue_info_id = v.id
        JOIN paper_authors pa ON p.id = pa.paper_id
        WHERE v.conference = 'NeurIPS' 
        AND v.year = 2025 
        AND v.track = 'Conference'
        AND p.status = 'accepted'
        GROUP BY pa.affiliation_country
        ORDER BY paper_count DESC
    """)
    
    countries = []
    india_stats = None
    for row in cursor.fetchall():
        country_data = {
            "affiliation_country": row[0],
            "paper_count": row[1],
            "author_count": row[2],
            "spotlights": 0,  # Not available without accept_type
            "orals": 0  # Not available without accept_type
        }
        countries.append(country_data)
        if row[0] == 'IN':
            india_stats = country_data.copy()
    
    # Get institution statistics for India
    cursor.execute("""
        SELECT 
            pa.affiliation_name,
            COUNT(DISTINCT p.id) as paper_count,
            COUNT(DISTINCT pa.author_id) as author_count
        FROM papers p
        JOIN venue_infos v ON p.venue_info_id = v.id
        JOIN paper_authors pa ON p.id = pa.paper_id
        WHERE v.conference = 'NeurIPS' 
        AND v.year = 2025 
        AND v.track = 'Conference'
        AND p.status = 'accepted'
        AND pa.affiliation_country = 'IN'
        GROUP BY pa.affiliation_name
        HAVING pa.affiliation_name IS NOT NULL AND pa.affiliation_name != ''
        ORDER BY paper_count DESC
        LIMIT 20
    """)
    
    institutions = []
    for row in cursor.fetchall():
        institutions.append({
            "institution": row[0],
            "paper_count": row[1],
            "author_count": row[2],
            "spotlights": 0,
            "orals": 0
        })
    
    # Build analytics JSON
    analytics = {
        "conferenceInfo": {
            "name": "NeurIPS",
            "year": 2025,
            "track": "Conference",
            "totalAcceptedPapers": total_papers,
            "totalAcceptedAuthors": total_authors
        },
        "globalStats": {
            "countries": countries
        },
        "indiaStats": india_stats or {
            "affiliation_country": "IN",
            "paper_count": 0,
            "author_count": 0,
            "spotlights": 0,
            "orals": 0
        },
        "topInstitutions": institutions,
        "note": "spotlights and orals data not available for NeurIPS 2025"
    }
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(analytics, f, indent=2)
    
    conn.close()
    print(f"Generated analytics at {output_path}")
    print(f"Total papers: {total_papers}, Total authors: {total_authors}")
    if india_stats:
        print(f"India: {india_stats['paper_count']} papers, {india_stats['author_count']} authors")

if __name__ == "__main__":
    db_path = "../indiaml/venues.db"
    output_path = "../ui/indiaml-tracker/public/tracker/neurips-2025-analytics.json"
    generate_analytics(db_path, output_path)
