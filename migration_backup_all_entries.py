#!/usr/bin/env python3
"""
Database migration script to backup ALL entries in the credits table with JSON backups,
regardless of their reviewed_status.
"""

import sqlite3
import json
import logging
from pathlib import Path
import sys

# Add the scripts_v3 directory to the path so we can import config
sys.path.append(str(Path(__file__).parent / "scripts_v3"))
from scripts_v3 import config

def setup_logging():
    """Setup logging for the migration script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def get_all_columns(cursor, table_name):
    """Get all column names for a table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return columns

def create_backup_json(cursor, row_data, columns):
    """Create a JSON backup of all row data."""
    backup_data = {}
    for i, column in enumerate(columns):
        backup_data[column] = row_data[i]
    return json.dumps(backup_data, ensure_ascii=False, indent=2)

def run_migration():
    """Run the database migration."""
    setup_logging()
    
    db_path = Path(config.DB_PATH)
    if not db_path.exists():
        logging.error(f"Database file not found: {db_path}")
        return False
    
    logging.info(f"Starting migration on database: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if the column already exists
        cursor.execute("PRAGMA table_info(credits)")
        existing_columns = [row[1] for row in cursor.fetchall()]
        
        if 'original_data_backup' in existing_columns:
            logging.info("Column 'original_data_backup' already exists.")
        else:
            # Add the new column
            logging.info("Adding 'original_data_backup' column to credits table...")
            cursor.execute("ALTER TABLE credits ADD COLUMN original_data_backup TEXT")
            logging.info("Column added successfully.")
        
        # Get all column names for the credits table
        columns = get_all_columns(cursor, 'credits')
        logging.info(f"Found {len(columns)} columns in credits table: {', '.join(columns)}")
        
        # Get total count of all entries
        cursor.execute("SELECT COUNT(*) FROM credits")
        total_entries = cursor.fetchone()[0]
        logging.info(f"Total entries in credits table: {total_entries}")
        
        # Get status distribution
        cursor.execute("SELECT reviewed_status, COUNT(*) FROM credits GROUP BY reviewed_status")
        status_counts = cursor.fetchall()
        logging.info("Current status distribution:")
        for status, count in status_counts:
            logging.info(f"  {status}: {count}")
        
        # Find all entries that don't have a backup yet
        cursor.execute("""
            SELECT * FROM credits 
            WHERE original_data_backup IS NULL OR original_data_backup = ''
        """)
        
        entries_to_backup = cursor.fetchall()
        logging.info(f"Found {len(entries_to_backup)} entries without backup data.")
        
        if entries_to_backup:
            # Create backups for all entries
            updated_count = 0
            for row_data in entries_to_backup:
                # Create JSON backup of all fields
                backup_json = create_backup_json(cursor, row_data, columns)
                
                # Update the row with the backup data
                cursor.execute("""
                    UPDATE credits 
                    SET original_data_backup = ? 
                    WHERE id = ?
                """, (backup_json, row_data[0]))  # Assuming 'id' is the first column
                
                updated_count += 1
                
                if updated_count % 50 == 0:
                    logging.info(f"Processed {updated_count}/{len(entries_to_backup)} entries...")
            
            logging.info(f"Successfully created backups for {updated_count} entries.")
        else:
            logging.info("All entries already have backup data.")
        
        # Commit the changes
        conn.commit()
        logging.info("Migration completed successfully!")
        
        # Verify the migration
        cursor.execute("""
            SELECT COUNT(*) FROM credits 
            WHERE original_data_backup IS NOT NULL 
            AND original_data_backup != ''
        """)
        backed_up_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM credits")
        total_count = cursor.fetchone()[0]
        
        logging.info(f"Verification: {backed_up_count}/{total_count} entries now have backup data.")
        
        # Show final status distribution
        cursor.execute("SELECT reviewed_status, COUNT(*) FROM credits GROUP BY reviewed_status")
        final_status_counts = cursor.fetchall()
        logging.info("Final status distribution:")
        for status, count in final_status_counts:
            logging.info(f"  {status}: {count}")
        
        conn.close()
        return True
        
    except Exception as e:
        logging.error(f"Migration failed: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False

if __name__ == "__main__":
    success = run_migration()
    if success:
        print("✅ Migration completed successfully!")
        sys.exit(0)
    else:
        print("❌ Migration failed!")
        sys.exit(1) 