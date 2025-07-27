#!/usr/bin/env python3
import sqlite3

def check_status():
    """Check the current status of credits in the database"""
    
    conn = sqlite3.connect('db/tvcredits_v3.db')
    cursor = conn.cursor()
    
    # Count credits with no assigned code
    cursor.execute('SELECT COUNT(*) FROM credits WHERE assigned_code IS NULL')
    no_code = cursor.fetchone()[0]
    print(f"Credits with no assigned code: {no_code}")
    
    # Count credits with no IMDB matches
    cursor.execute('SELECT COUNT(*) FROM credits WHERE imdb_matches IS NULL OR imdb_matches = ""')
    no_imdb = cursor.fetchone()[0]
    print(f"Credits with no IMDB matches: {no_imdb}")
    
    # Count by assignment status
    cursor.execute('SELECT code_assignment_status, COUNT(*) FROM credits GROUP BY code_assignment_status')
    status_counts = cursor.fetchall()
    print("\nCredits by assignment status:")
    for status, count in status_counts:
        print(f"  {status}: {count}")
    
    # Show some examples of credits with no IMDB matches
    cursor.execute('SELECT name, role_group_normalized FROM credits WHERE imdb_matches IS NULL OR imdb_matches = "" LIMIT 5')
    examples = cursor.fetchall()
    print(f"\nExamples of credits with no IMDB matches:")
    for name, role in examples:
        print(f"  {name} ({role})")
    
    conn.close()

if __name__ == "__main__":
    check_status() 