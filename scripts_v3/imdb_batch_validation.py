#!/usr/bin/env python3
"""
Step 4: IMDB Batch Validation with Code Assignment

This script processes all credits in the database and validates them against
the IMDB database, assigning nconst codes or internal gp codes based on
the profession matching logic.

This is designed to run after VLM OCR processing (step 3) and before
the review interface.

Author: Assistant
Date: July 2025
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
from scripts_v3.imdb_name_validation import IMDBNameValidator, CodeAssignmentStatus
from scripts_v3.utils import get_imdb_validator

class IMDBBatchValidatorWithCodeAssignment:
    """Batch processor for IMDB validation with code assignment."""
    
    def __init__(self):
        self.validator: Optional[IMDBNameValidator] = None
        self.stats = {
            'total_credits': 0,
            'persons_processed': 0,
            'companies_processed': 0,
            'auto_assigned_nconst': 0,
            'auto_assigned_internal': 0,
            'manual_required': 0,
            'ambiguous': 0,
            'errors': 0,
            'already_processed': 0,
            'found_in_imdb': 0,
            'not_found_in_imdb': 0,
            'companies_skipped': 0
        }
    
    def _get_validator(self) -> IMDBNameValidator:
        """Get the IMDB validator instance."""
        if self.validator is None:
            self.validator = IMDBNameValidator()
        return self.validator
    
    def get_unprocessed_credits(self, episode_id: Optional[str] = None, force_reprocess: bool = False) -> List[Dict[str, Any]]:
        """
        Get credits that haven't been processed for code assignment yet.
        
        Args:
            episode_id: If specified, only process credits for this episode
            force_reprocess: If True, reprocess all credits even if already assigned
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
                where_conditions.append("(assigned_code IS NULL OR code_assignment_status IN ('manual_required', 'ambiguous'))")
            
            where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
            
            query = f"""
                SELECT id, episode_id, name, normalized_name, role_group, role_group_normalized, is_person,
                       assigned_code, code_assignment_status
                FROM {config.DB_TABLE_CREDITS}
                {where_clause}
                ORDER BY episode_id, name
            """
            
            cursor.execute(query, params)
            credits = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            logging.info(f"Found {len(credits)} credits to process for code assignment")
            return credits
            
        except sqlite3.Error as e:
            logging.error(f"Database error getting unprocessed credits: {e}")
            return []
    
    def update_credit_with_code_assignment(self, credit_id: int, assigned_code: str, 
                                         assignment_status: CodeAssignmentStatus, 
                                         imdb_matches_json: Optional[str] = None) -> bool:
        """
        Update a credit with code assignment results.
        
        Args:
            credit_id: The ID of the credit to update
            assigned_code: The assigned code (nconst or internal gp code)
            assignment_status: The assignment status
            imdb_matches_json: JSON string of IMDB matches for ambiguous cases
        
        Returns:
            True if successful, False otherwise
        """        
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute(
                f"""UPDATE {config.DB_TABLE_CREDITS} 
                    SET assigned_code = ?, code_assignment_status = ?, imdb_matches = ?
                    WHERE id = ?""",
                (assigned_code, assignment_status.value, imdb_matches_json, credit_id)
            )
            
            conn.commit()
            conn.close()
            return True
            
        except sqlite3.Error as e:
            logging.error(f"Database error updating credit {credit_id}: {e}")
            return False
    
    def process_credit_with_code_assignment(self, credit: Dict[str, Any]) -> bool:
        """
        Process a single credit for code assignment.
        
        Args:
            credit: Credit dictionary with name, role_group, etc.
        
        Returns:
            True if processed successfully, False otherwise
        """
        name = credit.get('name', '')
        role_group = credit.get('role_group_normalized') or credit.get('role_group', '')
        is_person = credit.get('is_person')
        credit_id = credit.get('id')
        
        if not name:
            logging.warning(f"Credit {credit_id} has no name, skipping")
            return False
        
        try:
            validator = self._get_validator()
            
            # Validate and assign code
            result = validator.validate_name_with_code_assignment(name, role_group, is_person)
            
            # Determine if person was found in IMDB (regardless of assignment success)
            is_found_in_imdb = result.matches and len(result.matches) > 0
            
            # Update database with the result
            success = self.update_credit_with_code_assignment(
                credit_id,
                result.assigned_code,
                result.assignment_status,
                result.imdb_matches_json
            )
            
            if success:
                # Update statistics
                if result.assignment_status == CodeAssignmentStatus.AUTO_ASSIGNED:
                    if result.assigned_code and result.assigned_code.startswith('nm'):
                        self.stats['auto_assigned_nconst'] += 1
                    else:
                        self.stats['auto_assigned_internal'] += 1
                elif result.assignment_status == CodeAssignmentStatus.INTERNAL_ASSIGNED:
                    self.stats['auto_assigned_internal'] += 1
                elif result.assignment_status == CodeAssignmentStatus.MANUAL_REQUIRED:
                    self.stats['manual_required'] += 1
                elif result.assignment_status == CodeAssignmentStatus.AMBIGUOUS:
                    self.stats['ambiguous'] += 1
                
                # Track IMDB search results (only for persons, not companies)
                if is_person not in [False, 0]:
                    if result.matches and len(result.matches) > 0:
                        self.stats['found_in_imdb'] += 1
                    else:
                        self.stats['not_found_in_imdb'] += 1
                
                # Log the result
                status_msg = f"[{result.assignment_status.value.upper()}] '{name}' -> {result.assigned_code}"
                if result.validation_method:
                    status_msg += f" (method: {result.validation_method})"
                logging.info(status_msg)
                
                return True
            else:
                self.stats['errors'] += 1
                return False
            
        except Exception as e:
            logging.error(f"Error processing credit '{name}': {e}")
            self.stats['errors'] += 1
            return False
    
    def process_credits_batch(self, credits: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """
        Process a batch of credits for code assignment.
        
        Args:
            credits: List of credit dictionaries to process
            batch_size: Number of credits to process before showing progress
        """
        total_credits = len(credits)
        self.stats['total_credits'] = total_credits
        
        logging.info(f"Starting IMDB validation with code assignment for {total_credits} credits...")
        
        processed = 0
        for i, credit in enumerate(credits):
            name = credit.get('name', '')
            is_person = credit.get('is_person')
            
            # Skip if already processed with valid assignment (shouldn't happen if query is correct)
            if credit.get('assigned_code') and credit.get('code_assignment_status') not in ['manual_required', 'ambiguous']:
                self.stats['already_processed'] += 1
                continue
            
            # Process the credit
            success = self.process_credit_with_code_assignment(credit)
            
            if success:
                if is_person is False or is_person == 0:
                    self.stats['companies_processed'] += 1
                else:
                    self.stats['persons_processed'] += 1
            
            processed += 1
            
            # Show progress every batch_size items
            if processed % batch_size == 0 or processed == total_credits:
                progress_pct = (processed / total_credits) * 100
                logging.info(f"Progress: {processed}/{total_credits} ({progress_pct:.1f}%) - "
                           f"Auto nconst: {self.stats['auto_assigned_nconst']}, "
                           f"Auto internal: {self.stats['auto_assigned_internal']}, "
                           f"Manual required: {self.stats['manual_required']}, "
                           f"Ambiguous: {self.stats['ambiguous']}")
    
    def print_final_stats(self) -> None:
        """Print final statistics of the code assignment process."""
        stats = self.stats
        total = stats['total_credits']
        
        print("\n" + "="*60)
        print("🎯 IMDB CODE ASSIGNMENT COMPLETED")
        print("="*60)
        print(f"📊 Total credits processed: {total}")
        print(f"👤 Persons processed: {stats['persons_processed']}")
        print(f"🏢 Companies processed: {stats['companies_processed']}")
        print(f"✅ Auto-assigned nconst: {stats['auto_assigned_nconst']}")
        print(f"🔧 Auto-assigned internal: {stats['auto_assigned_internal']}")
        print(f"⚠️  Manual review required: {stats['manual_required']}")
        print(f"❓ Ambiguous cases: {stats['ambiguous']}")
        print(f"❌ Errors: {stats['errors']}")
        print(f"⏭️  Already processed: {stats['already_processed']}")
        
        # IMDB search statistics (for persons only)
        if stats['found_in_imdb'] > 0 or stats['not_found_in_imdb'] > 0:
            print(f"🎭 Found in IMDB: {stats['found_in_imdb']}")
            print(f"❓ Not found in IMDB: {stats['not_found_in_imdb']}")
            
            total_searched = stats['found_in_imdb'] + stats['not_found_in_imdb']
            if total_searched > 0:
                match_rate = (stats['found_in_imdb'] / total_searched) * 100
                print(f"📈 IMDB match rate: {match_rate:.1f}%")
        
        total_processed = stats['persons_processed'] + stats['companies_processed']
        if total_processed > 0:
            auto_rate = ((stats['auto_assigned_nconst'] + stats['auto_assigned_internal']) / total_processed) * 100
            print(f"� Automatic assignment rate: {auto_rate:.1f}%")
        
        print("="*60)


# Legacy class for backward compatibility
class IMDBBatchValidator(IMDBBatchValidatorWithCodeAssignment):
    """Legacy class that maps to the new code assignment validator."""
    
    def update_imdb_status(self, credit_id: int, is_present: bool) -> bool:
        """Legacy method for backward compatibility."""
        return self.update_credit_with_code_assignment(
            credit_id, 
            None,  # No code assigned in legacy mode
            CodeAssignmentStatus.INTERNAL_ASSIGNED if not is_present else CodeAssignmentStatus.AUTO_ASSIGNED,
            None
        )
    
    def validate_credit(self, credit: Dict[str, Any]) -> bool:
        """Legacy method for backward compatibility."""
        return self.process_credit_with_code_assignment(credit)


def main():
    """Main function to run IMDB batch validation with code assignment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="IMDB Batch Validation with Code Assignment - Step 4")
    parser.add_argument('--episode-id', type=str, help='Process only credits for this episode')
    parser.add_argument('--force-reprocess', action='store_true', 
                       help='Reprocess all credits even if already assigned')
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
            logging.FileHandler('imdb_code_assignment.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create batch validator
    batch_validator = IMDBBatchValidatorWithCodeAssignment()
    
    # Get credits to process
    credits = batch_validator.get_unprocessed_credits(
        episode_id=args.episode_id,
        force_reprocess=args.force_reprocess
    )
    
    if not credits:
        print("✅ No credits need code assignment processing.")
        return
    
    # Process the credits
    start_time = time.time()
    batch_validator.process_credits_batch(credits, batch_size=args.batch_size)
    end_time = time.time()
    
    # Print final statistics
    batch_validator.print_final_stats()
    
    processing_time = end_time - start_time
    print(f"⏱️  Processing time: {processing_time:.1f} seconds")
    
    if credits and processing_time > 0:
        credits_per_second = len(credits) / processing_time
        print(f"🚀 Processing speed: {credits_per_second:.1f} credits/second")
    elif credits:
        print(f"🚀 Processing speed: {len(credits)} credits processed instantly")


if __name__ == "__main__":
    main()
