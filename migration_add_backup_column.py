#!/usr/bin/env python3
"""
Database migration script to add original_data_backup column and populate it with JSON backups
of all existing 'kept' entries in the credits table.
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
            logging.info("Column 'original_data_backup' already exists. Skipping column creation.")
        else:
            # Add the new column
            logging.info("Adding 'original_data_backup' column to credits table...")
            cursor.execute("ALTER TABLE credits ADD COLUMN original_data_backup TEXT")
            logging.info("Column added successfully.")
        
        # Get all column names for the credits table
        columns = get_all_columns(cursor, 'credits')
        logging.info(f"Found {len(columns)} columns in credits table: {', '.join(columns)}")
        
        # Find all 'kept' entries that don't have a backup yet
        cursor.execute("""
            SELECT * FROM credits 
            WHERE reviewed_status = 'kept' 
            AND (original_data_backup IS NULL OR original_data_backup = '')
        """)
        
        kept_entries = cursor.fetchall()
        logging.info(f"Found {len(kept_entries)} 'kept' entries without backup data.")
        
        if kept_entries:
            # Create backups for all 'kept' entries
            updated_count = 0
            for row_data in kept_entries:
                # Create JSON backup of all fields
                backup_json = create_backup_json(cursor, row_data, columns)
                
                # Update the row with the backup data
                cursor.execute("""
                    UPDATE credits 
                    SET original_data_backup = ? 
                    WHERE id = ?
                """, (backup_json, row_data[0]))  # Assuming 'id' is the first column
                
                updated_count += 1
                
                if updated_count % 100 == 0:
                    logging.info(f"Processed {updated_count}/{len(kept_entries)} entries...")
            
            logging.info(f"Successfully created backups for {updated_count} entries.")
        else:
            logging.info("No 'kept' entries found that need backup creation.")
        
        # Commit the changes
        conn.commit()
        logging.info("Migration completed successfully!")
        
        # Verify the migration
        cursor.execute("""
            SELECT COUNT(*) FROM credits 
            WHERE reviewed_status = 'kept' 
            AND original_data_backup IS NOT NULL 
            AND original_data_backup != ''
        """)
        backed_up_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM credits WHERE reviewed_status = 'kept'")
        total_kept_count = cursor.fetchone()[0]
        
        logging.info(f"Verification: {backed_up_count}/{total_kept_count} 'kept' entries now have backup data.")
        
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