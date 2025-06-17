#!/usr/bin/env python3
"""
Step 4: IMDB Batch Validation

This script processes all credits in the database and validates them against
the IMDB database, setting the is_present_in_imdb column.

This is designed to run after VLM OCR processing (step 3) and before
the review interface.

Author: Assistant
Date: June 2025
"""

import sqlite3
import logging
import time
from typing import List, Dict, Any, Tuple, Optional
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts_v3.config as config
from scripts_v3.imdb_name_validation import IMDBNameValidator
from scripts_v3.utils import get_imdb_validator

class IMDBBatchValidator:
    """Batch processor for IMDB validation of credits."""
    
    def __init__(self):
        self.validator: Optional[IMDBNameValidator] = None
        self.stats = {
            'total_credits': 0,
            'persons_processed': 0,
            'companies_skipped': 0,
            'found_in_imdb': 0,
            'not_found_in_imdb': 0,
            'errors': 0,
            'already_processed': 0
        }
    
    def _get_validator(self) -> IMDBNameValidator:
        """Get the IMDB validator instance."""
        if self.validator is None:
            self.validator = get_imdb_validator()
        return self.validator
    
    def get_unprocessed_credits(self, episode_id: Optional[str] = None, force_reprocess: bool = False) -> List[Dict[str, Any]]:
        """
        Get credits that haven't been processed for IMDB validation yet.
        
        Args:
            episode_id: If specified, only process credits for this episode
            force_reprocess: If True, reprocess all credits even if already validated
          Returns:
            List of credit dictionaries
        """
        try:
            conn = sqlite3.connect(config.DB_PATH)
            conn.row_factory = sqlite3.Row  # This allows us to access columns by name
            cursor = conn.cursor()
            
            # Build the query based on parameters
            where_conditions = []
            params = []
            
            if episode_id:
                where_conditions.append("episode_id = ?")
                params.append(episode_id)
            
            if not force_reprocess:
                where_conditions.append("is_present_in_imdb IS NULL")
            
            where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
            
            query = f"""
                SELECT id, episode_id, name, normalized_name, role_group, is_person
                FROM {config.DB_TABLE_CREDITS}
                {where_clause}
                ORDER BY episode_id, name
            """
            
            cursor.execute(query, params)
            credits = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            logging.info(f"Found {len(credits)} credits to process for IMDB validation")
            return credits
            
        except sqlite3.Error as e:
            logging.error(f"Database error getting unprocessed credits: {e}")
            return []
    
    def update_imdb_status(self, credit_id: int, is_present: bool) -> bool:
        """
        Update the is_present_in_imdb status for a credit.
        
        Args:
            credit_id: The ID of the credit to update
            is_present: Whether the name was found in IMDB
        
        Returns:
            True if successful, False otherwise
        """        
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute(
                f"UPDATE {config.DB_TABLE_CREDITS} SET is_present_in_imdb = ? WHERE id = ?",
                (is_present, credit_id)
            )
            
            conn.commit()
            conn.close()
            return True
            
        except sqlite3.Error as e:
            logging.error(f"Database error updating IMDB status for credit {credit_id}: {e}")
            return False
    
    def validate_credit(self, credit: Dict[str, Any]) -> bool:
        """
        Validate a single credit against IMDB.
        
        Args:
            credit: Credit dictionary with name, normalized_name, etc.
        
        Returns:
            True if found in IMDB, False otherwise
        """
        name = credit.get('name', '')
        normalized_name = credit.get('normalized_name', '')
        role_group = credit.get('role_group', '')
        is_person = credit.get('is_person')
        
        if not name:
            logging.warning(f"Credit {credit.get('id')} has no name, skipping")
            return False
        
        try:
            validator = self._get_validator()
            
            # Use normalized name if available, otherwise compute it
            if normalized_name:
                result = validator.validate_name_with_normalized(name, normalized_name, role_group, is_person)
            else:
                result = validator.validate_name(name, role_group, is_person)
            
            is_valid = result.get('is_valid', False)
            validation_method = result.get('validation_method', 'unknown')
            
            # Log the result
            status = "FOUND" if is_valid else "NOT_FOUND"
            logging.info(f"[{status}] '{name}' (method: {validation_method})")
            
            return is_valid
            
        except Exception as e:
            logging.error(f"Error validating credit '{name}': {e}")
            self.stats['errors'] += 1
            return False
    
    def process_credits_batch(self, credits: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """
        Process a batch of credits for IMDB validation.
        
        Args:
            credits: List of credit dictionaries to process
            batch_size: Number of credits to process before showing progress
        """
        total_credits = len(credits)
        self.stats['total_credits'] = total_credits
        
        logging.info(f"Starting IMDB validation for {total_credits} credits...")
        
        processed = 0
        for i, credit in enumerate(credits):
            credit_id = credit['id']
            name = credit.get('name', '')
            is_person = credit.get('is_person')
            
            # Skip if already processed (shouldn't happen if query is correct)
            if credit.get('is_present_in_imdb') is not None:
                self.stats['already_processed'] += 1
                continue
            
            # Skip companies if is_person is False or 0
            if is_person is False or is_person == 0:
                # Companies are considered "valid" but not necessarily in IMDB
                # We set them as True to avoid flagging them as problematic
                success = self.update_imdb_status(credit_id, True)
                if success:
                    self.stats['companies_skipped'] += 1
                    logging.info(f"[COMPANY] '{name}' - marked as valid (company)")
                else:
                    self.stats['errors'] += 1
            else:
                # Process person names
                is_found = self.validate_credit(credit)
                success = self.update_imdb_status(credit_id, is_found)
                
                if success:
                    self.stats['persons_processed'] += 1
                    if is_found:
                        self.stats['found_in_imdb'] += 1
                    else:
                        self.stats['not_found_in_imdb'] += 1
                else:
                    self.stats['errors'] += 1
            
            processed += 1
            
            # Show progress every batch_size items
            if processed % batch_size == 0 or processed == total_credits:
                progress_pct = (processed / total_credits) * 100
                logging.info(f"Progress: {processed}/{total_credits} ({progress_pct:.1f}%) - "
                           f"Found: {self.stats['found_in_imdb']}, "
                           f"Not found: {self.stats['not_found_in_imdb']}, "
                           f"Companies: {self.stats['companies_skipped']}")
    
    def print_final_stats(self) -> None:
        """Print final statistics of the validation process."""
        stats = self.stats
        total = stats['total_credits']
        
        print("\n" + "="*60)
        print("🎯 IMDB BATCH VALIDATION COMPLETED")
        print("="*60)
        print(f"📊 Total credits processed: {total}")
        print(f"👤 Persons validated: {stats['persons_processed']}")
        print(f"🏢 Companies skipped: {stats['companies_skipped']}")
        print(f"✅ Found in IMDB: {stats['found_in_imdb']}")
        print(f"❌ Not found in IMDB: {stats['not_found_in_imdb']}")
        print(f"⚠️  Errors: {stats['errors']}")
        print(f"⏭️  Already processed: {stats['already_processed']}")
        
        if stats['persons_processed'] > 0:
            found_rate = (stats['found_in_imdb'] / stats['persons_processed']) * 100
            print(f"📈 IMDB match rate for persons: {found_rate:.1f}%")
        
        print("="*60)


def main():
    """Main function to run IMDB batch validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="IMDB Batch Validation - Step 4")
    parser.add_argument('--episode-id', type=str, help='Process only credits for this episode')
    parser.add_argument('--force-reprocess', action='store_true', 
                       help='Reprocess all credits even if already validated')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Number of credits to process before showing progress (default: 100)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Set the logging level')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('imdb_validation.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create batch validator
    batch_validator = IMDBBatchValidator()
    
    # Get credits to process
    credits = batch_validator.get_unprocessed_credits(
        episode_id=args.episode_id,
        force_reprocess=args.force_reprocess
    )
    
    if not credits:
        print("✅ No credits need IMDB validation processing.")
        return
    
    # Process the credits
    start_time = time.time()
    batch_validator.process_credits_batch(credits, batch_size=args.batch_size)
    end_time = time.time()
    
    # Print final statistics
    batch_validator.print_final_stats()
    
    processing_time = end_time - start_time
    print(f"⏱️  Processing time: {processing_time:.1f} seconds")
    
    if credits:
        credits_per_second = len(credits) / processing_time
        print(f"🚀 Processing speed: {credits_per_second:.1f} credits/second")


if __name__ == "__main__":
    main()
