#!/usr/bin/env python3
"""
Script to check the current distribution of code assignment statuses in the database
"""

import sqlite3
from pathlib import Path

def check_current_status():
    """Check the current distribution of code assignment statuses"""
    
    print("🔍 Checking current database status...")
    
    conn = sqlite3.connect('db/tvcredits_v3.db')
    cursor = conn.cursor()
    
    # Get total count
    cursor.execute("SELECT COUNT(*) FROM credits")
    total_credits = cursor.fetchone()[0]
    
    # Get distribution by code_assignment_status
    cursor.execute("""
        SELECT code_assignment_status, COUNT(*) as count
        FROM credits 
        GROUP BY code_assignment_status
        ORDER BY count DESC
    """)
    
    status_distribution = cursor.fetchall()
    
    # Get distribution by assigned_code type
    cursor.execute("""
        SELECT 
            CASE 
                WHEN assigned_code LIKE 'nm%' THEN 'IMDB codes'
                WHEN assigned_code LIKE 'gp%' THEN 'Person codes'
                WHEN assigned_code LIKE 'cm%' THEN 'Company codes'
                WHEN assigned_code IS NULL THEN 'No code'
                ELSE 'Other codes'
            END as code_type,
            COUNT(*) as count
        FROM credits 
        GROUP BY code_type
        ORDER BY count DESC
    """)
    
    code_type_distribution = cursor.fetchall()
    
    # Get credits that would be processed by the fix script
    cursor.execute("""
        SELECT COUNT(*) 
        FROM credits 
        WHERE (assigned_code IS NULL OR code_assignment_status IN ('manual_required', 'ambiguous') OR imdb_matches IS NULL OR imdb_matches = '')
    """)
    
    processable_credits = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"\n📊 Total credits in database: {total_credits:,}")
    print(f"📋 Credits that would be processed by fix script: {processable_credits:,}")
    
    print(f"\n🎯 Code Assignment Status Distribution:")
    for status, count in status_distribution:
        percentage = (count / total_credits) * 100
        print(f"  - {status or 'NULL'}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n🔢 Assigned Code Type Distribution:")
    for code_type, count in code_type_distribution:
        percentage = (count / total_credits) * 100
        print(f"  - {code_type}: {count:,} ({percentage:.1f}%)")
    
    if processable_credits == 0:
        print(f"\n✅ All credits already have codes assigned!")
        print("The fix script found 0 credits to process because all credits already have appropriate codes.")
    else:
        print(f"\n⚠️  There are {processable_credits} credits that could be processed.")

if __name__ == "__main__":
    check_current_status() 