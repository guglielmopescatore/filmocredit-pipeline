"""
IMDB Name Validation Module

This module handles:
1. Creating and indexing IMDB names database from name.basics.tsv
2. Validating extracted names against IMDB database
3. Flagging unrecognized names as problematic for human review
"""

import sqlite3
import csv
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from thefuzz import fuzz
from scripts_v3 import config


@dataclass
class IMDBPerson:
    """Represents a person from IMDB name.basics.tsv"""
    nconst: str
    primary_name: str
    birth_year: Optional[int]
    death_year: Optional[int]
    primary_profession: List[str]
    known_for_titles: List[str]


class IMDBNameValidator:
    """Handles IMDB name validation and database operations"""
    
    def __init__(self):
        self.imdb_db_path = config.IMDB_DB_PATH
        self.tsv_path = config.IMDB_TSV_PATH
        self.fuzzy_threshold = 85  # Minimum similarity score for fuzzy matching
        
    def create_imdb_database(self, force_rebuild: bool = False) -> bool:
        """
        Create and populate IMDB names database from TSV file.
        All names are stored in lowercase for case-insensitive matching.
        """
        if self.imdb_db_path.exists() and not force_rebuild:
            logging.info(f"IMDB database already exists at {self.imdb_db_path}")
            return True
        if not self.tsv_path.exists():
            logging.error(f"IMDB TSV file not found: {self.tsv_path}")
            return False
        try:
            config.DB_DIR.mkdir(parents=True, exist_ok=True)
            if force_rebuild and self.imdb_db_path.exists():
                self.imdb_db_path.unlink()
            conn = sqlite3.connect(self.imdb_db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS imdb_names (
                    nconst TEXT PRIMARY KEY,
                    primary_name TEXT NOT NULL,
                    primary_name_lower TEXT NOT NULL,
                    birth_year INTEGER,
                    death_year INTEGER,
                    primary_profession TEXT,
                    known_for_titles TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_primary_name_lower ON imdb_names (primary_name_lower)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_profession ON imdb_names (primary_profession)")
            logging.info(f"Starting to import IMDB data from {self.tsv_path}")
            with open(self.tsv_path, 'r', encoding='utf-8') as tsv_file:
                header = tsv_file.readline().strip().split('\t')
                logging.info(f"TSV header: {header}")
                batch_size = 10000
                batch_data = []
                total_processed = 0
                reader = csv.reader(tsv_file, delimiter='\t')
                for row in reader:
                    if len(row) >= 6:
                        nconst, primary_name, birth_year, death_year, primary_profession, known_for_titles = row
                        birth_year_int = int(birth_year) if birth_year != '\\N' else None
                        death_year_int = int(death_year) if death_year != '\\N' else None
                          # Apply the same cleaning logic used for validation
                        primary_name_lower = self._clean_name_for_validation(primary_name).lower().strip()
                        
                        # Skip entries that become empty after cleaning
                        if not primary_name_lower:
                            continue
                        
                        batch_data.append((
                            nconst,
                            primary_name,
                            primary_name_lower,
                            birth_year_int,
                            death_year_int,
                            primary_profession if primary_profession != '\\N' else '',
                            known_for_titles if known_for_titles != '\\N' else ''
                        ))
                        if len(batch_data) >= batch_size:
                            cursor.executemany("""
                                INSERT OR REPLACE INTO imdb_names 
                                (nconst, primary_name, primary_name_lower, birth_year, death_year, primary_profession, known_for_titles)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, batch_data)
                            total_processed += len(batch_data)
                            batch_data = []
                            if total_processed % 100000 == 0:
                                logging.info(f"Processed {total_processed:,} IMDB entries...")
                                conn.commit()
                if batch_data:
                    cursor.executemany("""
                        INSERT OR REPLACE INTO imdb_names 
                        (nconst, primary_name, primary_name_lower, birth_year, death_year, primary_profession, known_for_titles)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, batch_data)
                    total_processed += len(batch_data)
            conn.commit()
            cursor.execute("SELECT COUNT(*) FROM imdb_names")
            total_count = cursor.fetchone()[0]
            conn.close()
            logging.info(f"Successfully imported {total_count:,} IMDB names to database")
            return True
        except Exception as e:
            logging.error(f"Error creating IMDB database: {e}", exc_info=True)
            return False
    
    def _normalize_name(self, name: str) -> str:
        """
        Normalize a name for comparison.
        
        Args:
            name: Original name string
            
        Returns:
            Normalized name for comparison
        """
        if not name:
            return ""
            
        # Convert to lowercase and remove extra whitespace
        normalized = re.sub(r'\s+', ' ', name.lower().strip())
        
        # Remove common punctuation that might cause mismatches
        normalized = re.sub(r'[.,\-\'\"()"]', '', normalized)
        
        # Handle common name variations
        normalized = re.sub(r'\bjr\b', 'junior', normalized)
        normalized = re.sub(r'\bsr\b', 'senior', normalized)
        normalized = re.sub(r'\biii\b', '3', normalized)
        normalized = re.sub(r'\bii\b', '2', normalized)
        
        return normalized.strip()
    
    def _clean_name_for_validation(self, name: str) -> str:
        """
        Clean a name for validation by removing problematic characters and patterns.
        
        Args:
            name: Original name string
            
        Returns:
            Cleaned name suitable for validation
        """
        if not name:
            return ""
        
        import re
        
        # Remove multiple consecutive dashes or spaces
        cleaned = re.sub(r'[-\s]{2,}', ' ', name)
        
        # Remove single standalone dashes
        cleaned = re.sub(r'\s-\s', ' ', cleaned)
        cleaned = re.sub(r'^-\s*|\s*-$', '', cleaned)
        
        # Remove common OCR artifacts and noise
        cleaned = re.sub(r'[^\w\s\'-]', ' ', cleaned)
        
        # Remove very short words (likely OCR noise) except common initials
        words = cleaned.split()
        filtered_words = []
        for word in words:
            word = word.strip()
            if len(word) >= 2 or (len(word) == 1 and word.isupper()):
                filtered_words.append(word)
        
        cleaned = ' '.join(filtered_words)
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def validate_name(self, name: str, role_group: Optional[str] = None, is_person: Optional[bool] = None) -> Dict[str, Any]:
        """
        Validate a name against IMDB database using only exact (case-insensitive) matching.
        Tries all permutations of the space-separated words in the name.
        """
        import itertools        # Skip validation for company names as they won't be in IMDB names database
        logging.debug(f"IMDB validation called for name: '{name}', role_group: '{role_group}'")
        
        # Import the helper function for consistent company detection (fallback)
        from scripts_v3.utils import is_company_role_group
          # First check if we have explicit is_person information
        if is_person is False:
            logging.info(f"Skipping IMDB validation for company name: '{name}' (is_person=False)")
            return {
                'is_valid': True,  # Consider company names as valid (not problematic)
                'confidence': 1.0,
                'matches': [],
                'validation_method': 'company_name_skipped_is_person',
                'suggestion': None
            }
        # Fallback to role group detection for existing data
        elif is_company_role_group(role_group):
            logging.info(f"Skipping IMDB validation for company name: '{name}' (role: '{role_group}')")
            return {
                'is_valid': True,  # Consider company names as valid (not problematic)
                'confidence': 1.0,
                'matches': [],
                'validation_method': 'company_name_skipped_role_group',
                'suggestion': None
            }
        
        if not name or len(name.strip()) < 2:
            return {
                'is_valid': False,
                'confidence': 0.0,
                'matches': [],
                'validation_method': 'too_short',
                'suggestion': None
            }
        if not self.imdb_db_path.exists():
            logging.warning("IMDB database not found. Creating it now...")
            if not self.create_imdb_database():                return {
                    'is_valid': False,
                    'confidence': 0.0,
                    'matches': [],                    'validation_method': 'database_error',
                    'suggestion': None
                }
        
        try:
            conn = sqlite3.connect(self.imdb_db_path)
            cursor = conn.cursor()
            
            # Clean and validate the name before processing
            original_name = name
            cleaned_name = self._clean_name_for_validation(name)
            
            if not cleaned_name:
                logging.warning(f"Name '{original_name}' became empty after cleaning - skipping validation")
                conn.close()
                return {
                    'is_valid': False,
                    'confidence': 0.0,
                    'matches': [],
                    'validation_method': 'invalid_name_after_cleaning',
                    'suggestion': None
                }
              # Split into words and check for reasonable limits
            words = [w for w in cleaned_name.lower().strip().split() if w and len(w) > 1]
            
            # Safeguard: detect likely concatenated names (too many words)
            if len(words) > 6:
                logging.warning(f"Name '{original_name}' has {len(words)} words - likely concatenated multiple names, marking as invalid")
                conn.close()
                return {
                    'is_valid': False,
                    'confidence': 0.0,
                    'matches': [],
                    'validation_method': 'concatenated_names_detected',
                    'suggestion': 'This appears to be multiple names concatenated together - needs manual review'
                }
            
            # Safeguard: limit the number of words to prevent combinatorial explosion
            if len(words) > 4:
                logging.warning(f"Name '{original_name}' has {len(words)} words - limiting to first 4 words for validation")
                words = words[:4]
            
            if len(words) == 0:
                logging.warning(f"Name '{original_name}' has no valid words after processing")
                conn.close()
                return {
                    'is_valid': False,
                    'confidence': 0.0,
                    'matches': [],
                    'validation_method': 'no_valid_words',
                    'suggestion': None
                }
            
            # Generate permutations with a reasonable limit
            import math
            max_permutations = 24  # 4! = 24, reasonable limit
            if math.factorial(len(words)) > max_permutations:
                logging.warning(f"Name '{original_name}' would generate {math.factorial(len(words))} permutations - using only original order")
                permutations = [' '.join(words)]
            else:
                permutations = set([' '.join(p) for p in itertools.permutations(words)])
            
            logging.info(f"IMDB validation for '{original_name}' -> cleaned: '{cleaned_name}' -> words: {words} -> {len(permutations)} permutations")
            
            found_matches = []
            for perm in permutations:
                logging.debug(f"Searching IMDB for exact match: '{perm}'")
                cursor.execute(
                    "SELECT * FROM imdb_names WHERE primary_name_lower = ? LIMIT 5",
                    (perm,)
                )
                matches = cursor.fetchall()
                if matches:
                    logging.info(f"Found {len(matches)} IMDB match(es) for permutation '{perm}': {[m[1] for m in matches]}")
                    found_matches.extend(matches)
                else:
                    logging.debug(f"No IMDB matches found for permutation '{perm}'")
            
            if found_matches:
                formatted = self._format_matches(found_matches)
                best_match = self._select_best_match(formatted, role_group)
                conn.close()
                logging.info(f"IMDB validation SUCCESS for '{original_name}': found {len(found_matches)} match(es), best match: '{best_match['primary_name'] if best_match else 'N/A'}'")
                return {
                    'is_valid': True,
                    'confidence': 1.0,
                    'matches': formatted[:3],
                    'validation_method': 'exact_permutation_match',
                    'suggestion': best_match['primary_name'] if best_match else None
                }
            conn.close()
            logging.info(f"IMDB validation FAILED for '{original_name}': no matches found for any permutation")
            return {
                'is_valid': False,
                'confidence': 0.0,
                'matches': [],
                'validation_method': 'no_match',
                'suggestion': None
            }
        except Exception as e:
            logging.error(f"Error validating name '{name}': {e}", exc_info=True)
            return {
                'is_valid': False,                
                'confidence': 0.0,
                'matches': [],
                'validation_method': 'error',
                'suggestion': None
            }
    
    def _format_matches(self, raw_matches: List[Tuple]) -> List[Dict[str, Any]]:
        """Format raw database matches into dictionaries"""
        return [self._format_match(match) for match in raw_matches]
    
    def _format_match(self, raw_match: Tuple) -> Dict[str, Any]:
        """Format a single raw database match into a dictionary"""
        return {
            'nconst': raw_match[0],
            'primary_name': raw_match[1],
            'primary_name_lower': raw_match[2],
            'birth_year': raw_match[3],
            'death_year': raw_match[4],
            'primary_profession': raw_match[5],
            'known_for_titles': raw_match[6]
        }
    
    def _select_best_match(self, matches: List[Dict[str, Any]], role_group: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Select the best match based on role group context.
        
        Args:
            matches: List of potential matches
            role_group: Role group for context
            
        Returns:
            Best match or None
        """
        if not matches:
            return None
            
        if not role_group:
            return matches[0]  # Return first match if no role context
        
        # Map role groups to IMDB professions
        role_to_profession = {
            'cast': ['actor', 'actress'],
            'directors': ['director'],
            'writers': ['writer'],
            'producers': ['producer'],
            'composers': ['composer', 'music_department'],
            'cinematography': ['cinematographer'],
            'production_design': ['production_designer'],
            'costume_design': ['costume_designer'],
            'makeup': ['make_up_department'],
            'sound': ['sound_department'],
            'visual_effects': ['visual_effects']
        }
        
        target_professions = role_to_profession.get(role_group.lower(), [])
        
        if not target_professions:
            return matches[0]
        
        # Find matches with relevant professions
        for match in matches:
            professions = match.get('primary_profession', '').lower()
            if any(prof in professions for prof in target_professions):
                return match
        
        # If no profession match, return first match
        return matches[0]
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the IMDB database.
        
        Returns:
            Dictionary with database statistics
        """
        if not self.imdb_db_path.exists():
            return {'error': 'Database not found'}
        
        try:
            conn = sqlite3.connect(self.imdb_db_path)
            cursor = conn.cursor()
            
            # Total count
            cursor.execute("SELECT COUNT(*) FROM imdb_names")
            total_count = cursor.fetchone()[0]
            
            # Count by profession categories
            profession_stats = {}
            common_professions = ['actor', 'actress', 'director', 'writer', 'producer', 'composer']
            
            for profession in common_professions:
                cursor.execute(
                    "SELECT COUNT(*) FROM imdb_names WHERE primary_profession LIKE ?",
                    (f"%{profession}%",)
                )
                profession_stats[profession] = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_names': total_count,
                'profession_breakdown': profession_stats,
                'database_path': str(self.imdb_db_path),
                'database_size_mb': round(self.imdb_db_path.stat().st_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            logging.error(f"Error getting database stats: {e}", exc_info=True)
            return {'error': str(e)}


def validate_credits_batch(episode_id: str, max_invalid_names: int = 50) -> List[Dict[str, Any]]:
    """
    Validate all names in credits for an episode and return problematic ones.
    
    Args:
        episode_id: Episode to validate
        max_invalid_names: Maximum number of invalid names to return
        
    Returns:
        List of problematic credits with validation info
    """
    validator = IMDBNameValidator()
    problematic_credits = []
    
    try:
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        
        # Get all credits for the episode
        cursor.execute(f"""
            SELECT id, role_group_normalized, name, role_detail, 
                   source_frame, original_frame_number, scene_position, reviewed_status
            FROM {config.DB_TABLE_CREDITS} 
            WHERE episode_id = ? AND (reviewed_status IS NULL OR reviewed_status != 'kept')
            ORDER BY role_group_normalized, name
        """, (episode_id,))
        
        credits_data = cursor.fetchall()
        conn.close()
        
        if not credits_data:
            return []
        
        logging.info(f"Validating {len(credits_data)} credits for episode {episode_id}")
        
        processed_names = set()  # Avoid duplicate validation
        
        for credit in credits_data:
            credit_id, role_group, name, role_detail, source_frame, frame_numbers, scene_pos, reviewed_status = credit
            
            # Skip if we've already processed this name
            if name in processed_names:
                continue
            
            processed_names.add(name)
            
            # Validate the name
            validation_result = validator.validate_name(name, role_group)
            
            # If name is not valid or has low confidence, add to problematic list
            if not validation_result['is_valid'] or validation_result['confidence'] < 0.8:
                problematic_credit = {
                    'id': credit_id,
                    'episode_id': episode_id,
                    'role_group': role_group,
                    'name': name,
                    'role_detail': role_detail,
                    'source_frame': source_frame,
                    'frame_numbers': frame_numbers,
                    'scene_position': scene_pos,
                    'validation_result': validation_result,
                    'problem_types': ['invalid_imdb_name'],
                    'priority_score': 50 + (50 * (1 - validation_result['confidence']))  # Higher score for lower confidence
                }
                
                problematic_credits.append(problematic_credit)
                
                if len(problematic_credits) >= max_invalid_names:
                    break
        
        # Sort by priority score (lowest confidence first)
        problematic_credits.sort(key=lambda x: x['priority_score'], reverse=True)
        
        logging.info(f"Found {len(problematic_credits)} credits with invalid/uncertain IMDB names for episode {episode_id}")
        return problematic_credits
        
    except Exception as e:
        logging.error(f"Error validating credits batch for episode {episode_id}: {e}", exc_info=True)
        return []


def rebuild_imdb_database() -> bool:
    """
    Utility function to rebuild the IMDB database.
    
    Returns:
        True if successful
    """
    validator = IMDBNameValidator()
    return validator.create_imdb_database(force_rebuild=True)


def initialize_imdb_database() -> bool:
    """
    Initialize IMDB database if it doesn't exist.
    
    Returns:
        True if database exists or was created successfully
    """
    validator = IMDBNameValidator()
    return validator.create_imdb_database(force_rebuild=False)
