#!/usr/bin/env python3
"""
Export SQLite database tables to CSV files.

This script exports the FilmOCredit database tables (episodes and credits)
to CSV files for easy analysis and sharing.

Usage:
    python export_db_to_csv.py
    python export_db_to_csv.py --output-dir exports
    python export_db_to_csv.py --table credits
"""

import argparse
import csv
import sqlite3
import sys
from pathlib import Path
from datetime import datetime


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent


def get_db_path() -> Path:
    """Get the database path."""
    return get_project_root() / 'db' / 'tvcredits_v3.db'


def export_table_to_csv(
    db_path: Path,
    table_name: str,
    output_dir: Path,
    include_timestamp: bool = True,
    columns: list[str] | None = None,
) -> Path:
    """
    Export a database table to CSV file.
    
    Args:
        db_path: Path to SQLite database file
        table_name: Name of the table to export
        output_dir: Directory where CSV will be saved
        include_timestamp: Whether to include timestamp in filename
        
    Returns:
        Path to the created CSV file
        
    Raises:
        FileNotFoundError: If database doesn't exist
        sqlite3.Error: If database or table access fails
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
    filename_parts = [table_name]
    if timestamp:
        filename_parts.append(timestamp)
    filename = "_".join(filename_parts) + ".csv"
    output_path = output_dir / filename
    
    # Connect and export
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Build query depending on requested columns
        if columns:
            # Sanitize column names by joining; assume trusted internal use
            cols_sql = ", ".join(columns)
            query = f"SELECT {cols_sql} FROM {table_name}"
        else:
            query = f"SELECT * FROM {table_name}"

        cursor.execute(query)
        rows = cursor.fetchall()

        # Get column names
        column_names = [description[0] for description in cursor.description]
        
        # Post-process rows for credits table to clean episode_id
        if table_name == 'credits' and 'episode_id' in column_names:
            episode_id_idx = column_names.index('episode_id')
            processed_rows = []
            
            for row in rows:
                row_list = list(row)
                episode_id = row_list[episode_id_idx]
                
                # Remove _End and _Opening suffixes
                if episode_id:
                    episode_id = str(episode_id)
                    if episode_id.endswith('_End'):
                        episode_id = episode_id[:-4]  # Remove last 4 chars
                    elif episode_id.endswith('_Opening'):
                        episode_id = episode_id[:-8]  # Remove last 8 chars
                    row_list[episode_id_idx] = episode_id
                
                processed_rows.append(tuple(row_list))
            
            rows = processed_rows

        # Write to CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(column_names)  # Header
            writer.writerows(rows)  # Data

        print(f"✅ Exported {len(rows)} rows from '{table_name}' to: {output_path}")
        return output_path

    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error exporting table '{table_name}': {e}")
    finally:
        conn.close()


def list_tables(db_path: Path) -> list[str]:
    """
    List all tables in the database.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        List of table names
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        return tables
    finally:
        conn.close()


def get_table_info(db_path: Path, table_name: str) -> dict:
    """
    Get information about a table.
    
    Args:
        db_path: Path to SQLite database file
        table_name: Name of the table
        
    Returns:
        Dictionary with table information
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        
        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        return {
            'name': table_name,
            'row_count': row_count,
            'columns': [{'name': col[1], 'type': col[2]} for col in columns]
        }
    finally:
        conn.close()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Export FilmOCredit database tables to CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Export all tables:
    python export_db_to_csv.py
    
  Export to specific directory:
    python export_db_to_csv.py --output-dir exports
    
  Export only credits table:
    python export_db_to_csv.py --table credits
    
  Export without timestamp in filename:
    python export_db_to_csv.py --no-timestamp
    
  List available tables:
    python export_db_to_csv.py --list-tables
        """
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='exports',
        help='Output directory for CSV files (default: exports/)'
    )
    
    parser.add_argument(
        '--table',
        type=str,
        help='Export only specific table (default: export all tables)'
    )
    
    parser.add_argument(
        '--no-timestamp',
        action='store_true',
        help='Do not include timestamp in output filename'
    )
    
    parser.add_argument(
        '--list-tables',
        action='store_true',
        help='List all available tables and exit'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show detailed information about tables'
    )
    
    args = parser.parse_args()
    
    # Get paths
    db_path = get_db_path()
    output_dir = get_project_root() / args.output_dir
    
    # Check if database exists
    if not db_path.exists():
        print(f"❌ Error: Database not found at {db_path}")
        print("   Make sure you've run the pipeline at least once to create the database.")
        sys.exit(1)
    
    try:
        # List tables if requested
        if args.list_tables:
            tables = list_tables(db_path)
            print(f"\n📊 Available tables in database ({len(tables)}):")
            for table in tables:
                info = get_table_info(db_path, table)
                print(f"  • {table} ({info['row_count']} rows)")
            sys.exit(0)
        
        # Show detailed info if requested
        if args.info:
            tables = list_tables(db_path)
            print(f"\n📊 Database Information:")
            print(f"   Location: {db_path}")
            print(f"   Tables: {len(tables)}\n")
            
            for table in tables:
                info = get_table_info(db_path, table)
                print(f"  📋 Table: {info['name']}")
                print(f"     Rows: {info['row_count']}")
                print(f"     Columns ({len(info['columns'])}):")
                for col in info['columns']:
                    print(f"       - {col['name']} ({col['type']})")
                print()
            sys.exit(0)
        
        # Determine which tables to export
        if args.table:
            tables_to_export = [args.table]
        else:
            # Export main tables by default
            tables_to_export = ['episodes', 'credits', 'progressive_codes']
        
        # Export tables
        print(f"\n🔄 Exporting database to CSV...")
        print(f"   Database: {db_path}")
        print(f"   Output directory: {output_dir}\n")
        
        exported_files = []
        for table in tables_to_export:
            try:
                # Define specific columns for credits table to export all relevant fields
                columns = None
                if table == 'credits':
                    # Export all important columns including new imdb_name field
                    columns = [
                        'id',
                        'episode_id',
                        'source_frame',
                        'role_group',
                        'role_group_normalized',
                        'role_detail',
                        'name',
                        'imdb_name',
                        'normalized_name',
                        'is_person',
                        'assigned_code',
                        'code_assignment_status',
                        'imdb_matches',
                        'scene_position',
                        'original_frame_number',
                        'reviewed_status'
                    ]
                
                output_path = export_table_to_csv(
                    db_path,
                    table,
                    output_dir,
                    include_timestamp=not args.no_timestamp,
                    columns=columns
                )
                exported_files.append(output_path)
            except sqlite3.Error as e:
                print(f"⚠️  Warning: Could not export '{table}': {e}")
                continue
        
        # Summary
        print(f"\n✨ Export complete! Exported {len(exported_files)} file(s):")
        for filepath in exported_files:
            print(f"   📄 {filepath.relative_to(get_project_root())}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Export cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
