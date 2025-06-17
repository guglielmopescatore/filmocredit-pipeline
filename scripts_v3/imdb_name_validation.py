"""
IMDB Name Validation Module

This module handles:
1. Loading IMDB names database from name.basics.parquet using pandas DataFrame
2. Validating extracted names against IMDB database using normalized names
3. Flagging unrecognized names as problematic for human review
4. Name cleaning to handle OCR artifacts and malformed data
"""

import itertools
import logging
import math
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from scripts_v3 import config
# Note: normalize_name imported locally to avoid circular imports


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
    """Handles IMDB name validation using pandas DataFrame for performance"""
    
    def __init__(self):
        self.parquet_path = config.IMDB_PARQUET_PATH
        self.tsv_path = config.IMDB_TSV_PATH  # Fallback for backward compatibility
        self.fuzzy_threshold = 85  # Minimum similarity score for fuzzy matching
        self._imdb_df = None
        self._name_lookup = None
        self._initialized = False
        
    def _initialize_imdb_data(self) -> bool:
        """
        Initialize IMDB data from Parquet file using pandas DataFrame.
        Uses preprocessed data with normalizedName column for fast validation.
        Falls back to TSV format if Parquet is not available.
        """
        if self._initialized:
            return True
            
        # Check cache first
        cache_key = "imdb_initialized"
        if st.session_state.get(cache_key, False):
            self._initialized = True
            self._imdb_df = st.session_state.get("imdb_dataframe")
            self._name_lookup = st.session_state.get("imdb_name_lookup")
            if self._imdb_df is not None and self._name_lookup is not None:
                logging.info("IMDB database loaded from cache")
                return True
        
        # Try to load Parquet file first, then fall back to TSV
        data_path = None
        use_parquet = False
        if self.parquet_path.exists():
            data_path = self.parquet_path
            use_parquet = True
            logging.info(f"Using Parquet format: {data_path}")
        elif self.tsv_path.exists():
            # TSV exists but Parquet doesn't - create Parquet file automatically
            logging.info(f"Parquet not found but TSV exists. Creating Parquet file from {self.tsv_path}...")
            try:
                self._create_parquet_from_tsv()
                if self.parquet_path.exists():
                    data_path = self.parquet_path
                    use_parquet = True
                    logging.info(f"Successfully created and will use Parquet format: {data_path}")
                else:
                    # Fallback to TSV if Parquet creation failed
                    data_path = self.tsv_path
                    use_parquet = False
                    logging.warning(f"Parquet creation failed, falling back to TSV format: {data_path}")
            except Exception as e:
                logging.error(f"Failed to create Parquet file: {e}")
                data_path = self.tsv_path
                use_parquet = False
                logging.info(f"Falling back to TSV format: {data_path}")
        else:
            logging.error(f"Neither Parquet ({self.parquet_path}) nor TSV ({self.tsv_path}) file found")
            return False
            
        try:
            logging.info(f"Loading IMDB data from {data_path}...")
            start_time = time.time()
            
            if use_parquet:
                # Load the Parquet file
                self._imdb_df = pd.read_parquet(data_path)                # Check if normalizedName column exists
                if 'normalizedName' in self._imdb_df.columns:
                    logging.info("Using existing normalizedName column from Parquet file")
                    # Rename column to match expected name in queries
                    self._imdb_df = self._imdb_df.rename(columns={'normalizedName': 'normalized_name'})
                    # Use the DataFrame directly
                    self._name_lookup = self._imdb_df
                else:
                    logging.warning("normalizedName column not found in Parquet file, creating normalized names")
                    # Import normalize_name locally to avoid circular imports
                    from scripts_v3.utils import normalize_name
                    
                    # Fallback to creating normalized names
                    self._imdb_df['normalized_name'] = (
                        self._imdb_df['primaryName']
                        .fillna('')
                        .apply(normalize_name)
                    )
                    # Use the DataFrame directly
                    self._name_lookup = self._imdb_df
            else:
                # Load the TSV file (legacy format)
                self._imdb_df = pd.read_csv(
                    data_path,
                    sep='\t',
                    dtype={
                        'nconst': 'string',
                        'primaryName': 'string',
                        'birthYear': 'string',
                        'deathYear': 'string',
                        'primaryProfession': 'string',
                        'knownForTitles': 'string'
                    },
                    na_values=['\\N']
                )
                  # Create normalized names using utils.normalize_name
                logging.info("Creating normalized names using normalize_name function")
                self._imdb_df['normalized_name'] = (
                    self._imdb_df['primaryName']
                    .fillna('')
                    .apply(normalize_name)
                )
                
                # Use the DataFrame directly
                self._name_lookup = self._imdb_df
            
            # Remove empty normalized names
            self._imdb_df = self._imdb_df[self._imdb_df['normalized_name'] != '']
            
            load_time = time.time() - start_time
            logging.info(f"Successfully loaded {len(self._imdb_df):,} IMDB names in {load_time:.2f} seconds")
            
            # Cache the data in session state
            st.session_state[cache_key] = True
            st.session_state["imdb_dataframe"] = self._imdb_df
            st.session_state["imdb_name_lookup"] = self._name_lookup
            
            self._initialized = True
            return True
            
        except Exception as e:
            logging.error(f"Error loading IMDB data: {e}", exc_info=True)
            return False
    
    def _clean_name_for_validation(self, name: str) -> str:
        """
        Clean a name for validation by removing problematic characters and patterns.
        
        Args:
            name: Original name string
            
        Returns:
            Cleaned name suitable for validation
        """
        if not name or pd.isna(name):
            return ""
        
        name = str(name)
        
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
        Validate a name against IMDB database using exact (case-insensitive) matching.
        Tries all permutations of the space-separated words in the name.
        
        Args:
            name: Name to validate
            role_group: Role group for context (used for fallback company detection)
            is_person: Explicit flag if name refers to a person (True) or company (False)
            
        Returns:
            Dictionary with validation results
        """
        logging.debug(f"IMDB validation called for name: '{name}', role_group: '{role_group}', is_person: {is_person}")
        
        # Import the helper function for consistent company detection (fallback)
        from scripts_v3.utils import is_company_role_group
        
        # First check if we have explicit is_person information
        if is_person is False or is_person == 0:
            logging.info(f"Skipping IMDB validation for company name: '{name}' (is_person={is_person})")
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
        
        # Initialize IMDB data if needed
        if not self._initialize_imdb_data():
            return {
                'is_valid': False,
                'confidence': 0.0,
                'matches': [],
                'validation_method': 'database_error',
                'suggestion': None            }
        
        try:
            # Import normalize_name locally to avoid circular imports
            from scripts_v3.utils import normalize_name
            
            # Normalize the name using the same function used for IMDB data
            original_name = name
            normalized_name = normalize_name(name)
            
            if not normalized_name:
                logging.warning(f"Name '{original_name}' became empty after normalization - skipping validation")
                return {
                    'is_valid': False,
                    'confidence': 0.0,
                    'matches': [],
                    'validation_method': 'invalid_name_after_normalization',
                    'suggestion': None
                }
              # Split normalized name into words for permutation testing
            words = normalized_name.split()
            if len(words) == 0:
                logging.warning(f"No valid words found in normalized name '{normalized_name}'")
                return {
                    'is_valid': False,
                    'confidence': 0.0,
                    'matches': [],
                    'validation_method': 'no_valid_words',
                    'suggestion': None
                }
            
            # Safeguard: detect likely concatenated names (too many words)
            if len(words) > 6:
                logging.warning(f"Name '{original_name}' has {len(words)} words - likely concatenated multiple names, marking as invalid")
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
                return {
                    'is_valid': False,
                    'confidence': 0.0,
                    'matches': [],
                    'validation_method': 'no_valid_words',
                    'suggestion': None
                }
            
            # Generate permutations with a reasonable limit
            max_permutations = 24  # 4! = 24, reasonable limit
            if math.factorial(len(words)) > max_permutations:
                logging.warning(f"Name '{original_name}' would generate {math.factorial(len(words))} permutations - using only original order")
                permutations = [' '.join(words)]
            else:
                permutations = set([' '.join(p) for p in itertools.permutations(words)])
            
            logging.info(f"IMDB validation for '{original_name}' -> normalized: '{normalized_name}' -> words: {words} -> {len(permutations)} permutations")
              # Check each permutation against the lookup DataFrame
            found_matches = []
            for perm in permutations:
                logging.debug(f"Searching IMDB for exact match: '{perm}'")
                
                try:
                    # Search using the normalized_name column
                    matches = self._name_lookup[self._name_lookup['normalized_name'] == perm]
                    if not matches.empty:
                        logging.info(f"Found {len(matches)} IMDB match(es) for permutation '{perm}'")
                        found_matches.extend(matches.to_dict('records'))
                    else:
                        logging.debug(f"No IMDB matches found for permutation '{perm}'")
                except Exception as e:
                    logging.error(f"Error searching IMDB for permutation '{perm}': {e}")
                    continue
            
            if found_matches:
                formatted = self._format_matches(found_matches)
                best_match = self._select_best_match(formatted, role_group)
                logging.info(f"IMDB validation SUCCESS for '{original_name}': found {len(found_matches)} match(es)")                
                return {
                    'is_valid': True,
                    'confidence': 1.0,
                    'matches': formatted[:3],
                    'validation_method': 'exact_permutation_match',
                    'suggestion': best_match.get('primaryName') or best_match.get('normalizedName') if best_match else None
                }
            
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
    
    def validate_name_with_normalized(self, name: str, normalized_name: Optional[str] = None, 
                                     role_group: Optional[str] = None, is_person: Optional[bool] = None) -> Dict[str, Any]:
        """
        Validate a name against IMDB database, optionally using a pre-computed normalized name.
        This is an optimized version that can skip normalization if normalized_name is provided.
        
        Args:
            name: Original name to validate
            normalized_name: Pre-computed normalized name (if available)
            role_group: Role group for context (used for fallback company detection)
            is_person: Explicit flag if name refers to a person (True) or company (False)
            
        Returns:
            Dictionary with validation results
        """
        logging.debug(f"IMDB validation called for name: '{name}', normalized: '{normalized_name}', role_group: '{role_group}', is_person: {is_person}")
          # Import the helper function for consistent company detection (fallback)
        from scripts_v3.utils import is_company_role_group
        
        # First check if we have explicit is_person information
        if is_person is False or is_person == 0:
            logging.info(f"Skipping IMDB validation for company name: '{name}' (is_person={is_person})")
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
                'is_valid': True,
                'confidence': 1.0,
                'matches': [],
                'validation_method': 'company_name_skipped_role_group',
                'suggestion': None
            }
        
        # Use pre-computed normalized name if available, otherwise compute it
        if normalized_name:
            from scripts_v3.utils import normalize_name
            computed_normalized = normalize_name(name)
            if normalized_name != computed_normalized:
                logging.warning(f"Pre-computed normalized name '{normalized_name}' differs from computed '{computed_normalized}' for name '{name}'")
            normalized_to_search = normalized_name
        else:
            from scripts_v3.utils import normalize_name
            normalized_to_search = normalize_name(name)
        
        # Continue with the existing validation logic using the normalized name
        return self._validate_normalized_name(name, normalized_to_search, role_group, is_person)

    def _validate_normalized_name(self, original_name: str, normalized_name: str, 
                                 role_group: Optional[str] = None, is_person: Optional[bool] = None) -> Dict[str, Any]:
        """
        Internal method to validate using a normalized name.
        Separated for reuse by both validate_name and validate_name_with_normalized.
        """
        try:
            if not self._initialized:
                if not self._initialize_imdb_data():
                    return {
                        'is_valid': False,
                        'confidence': 0.0,
                        'matches': [],
                        'validation_method': 'initialization_failed',
                        'suggestion': None
                    }
            
            # Check for concatenated multiple names
            words = normalized_name.split()
            if len(words) > 6:
                logging.warning(f"Name '{original_name}' has {len(words)} words - likely concatenated multiple names, marking as invalid")
                return {
                    'is_valid': False,
                    'confidence': 0.0,
                    'matches': [],
                    'validation_method': 'concatenated_names_detected',
                    'suggestion': 'This appears to be multiple names concatenated together - needs manual review'
                }
            
            # Limit to first 4 words for practical search performance
            if len(words) > 4:
                logging.warning(f"Name '{original_name}' has {len(words)} words - limiting to first 4 words for validation")
                words = words[:4]
                normalized_name = ' '.join(words)
            
            # Generate permutations of the words and search
            import itertools
            permutations = list(itertools.permutations(words))
            total_perms = len(permutations)
            logging.info(f"IMDB validation for '{original_name}' -> normalized: '{normalized_name}' -> words: {words} -> {total_perms} permutations")
            
            found_matches = []
            for perm in permutations:
                perm_str = ' '.join(perm)
                logging.debug(f"Searching IMDB for exact match: '{perm_str}'")
                
                try:
                    matches = self._name_lookup[self._name_lookup['normalized_name'] == perm_str]
                    if not matches.empty:
                        logging.info(f"Found {len(matches)} IMDB match(es) for permutation '{perm_str}'")
                        found_matches.extend(matches.to_dict('records'))
                        # Stop at first successful permutation
                        break
                    else:
                        logging.debug(f"No IMDB matches found for permutation '{perm_str}'")
                except Exception as e:
                    logging.error(f"Error searching IMDB for permutation '{perm_str}': {e}")
                    continue
            
            if found_matches:
                formatted = self._format_matches(found_matches)
                best_match = self._select_best_match(formatted, role_group)
                logging.info(f"IMDB validation SUCCESS for '{original_name}': found {len(found_matches)} match(es)")
                return {
                    'is_valid': True,
                    'confidence': 1.0,
                    'matches': formatted[:3],
                    'validation_method': 'exact_permutation_match',
                    'suggestion': best_match.get('primaryName') if best_match and isinstance(best_match, dict) else None
                }
            
            logging.info(f"IMDB validation FAILED for '{original_name}': no matches found for any permutation")
            return {
                'is_valid': False,
                'confidence': 0.0,
                'matches': [],
                'validation_method': 'no_match',
                'suggestion': None
            }
            
        except Exception as e:
            logging.error(f"Error validating name '{original_name}': {e}", exc_info=True)
            return {
                'is_valid': False,
                'confidence': 0.0,
                'matches': [],
                'validation_method': 'error',
                'suggestion': None
            }
    
    def _format_matches(self, raw_matches: List[Dict]) -> List[Dict[str, Any]]:
        """Format raw DataFrame matches into dictionaries"""
        formatted = []
        for match in raw_matches:
            # Convert NaN values to None and handle data types
            formatted_match = {}
            for key, value in match.items():
                if pd.isna(value):
                    formatted_match[key] = None
                elif key in ['birthYear', 'deathYear']:
                    try:
                        formatted_match[key] = int(value) if value else None
                    except (ValueError, TypeError):
                        formatted_match[key] = None
                else:
                    formatted_match[key] = str(value) if value else ''
            formatted.append(formatted_match)
        return formatted
    
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
            professions = str(match.get('primaryProfession', '')).lower()
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
        if not self._initialize_imdb_data():
            return {'error': 'Database initialization failed'}
        
        try:
            total_count = len(self._imdb_df)
            
            # Count by profession categories
            profession_stats = {}
            common_professions = ['actor', 'actress', 'director', 'writer', 'producer', 'composer']
            
            for profession in common_professions:
                count = self._imdb_df['primaryProfession'].fillna('').str.contains(profession, case=False).sum()
                profession_stats[profession] = int(count)
            
            return {
                'total_names': total_count,
                'profession_breakdown': profession_stats,
                'tsv_path': str(self.tsv_path),
                'tsv_size_mb': round(self.tsv_path.stat().st_size / (1024 * 1024), 2) if self.tsv_path.exists() else 0
            }
            
        except Exception as e:
            logging.error(f"Error getting database stats: {e}", exc_info=True)
            return {'error': str(e)}


    def _create_parquet_from_tsv(self) -> None:
        """
        Create Parquet file from TSV file using the same normalization as the main codebase.
        This ensures consistent normalization between IMDB data and credits data.
        """
        import pandas as pd
        # Import normalize_name locally to avoid circular imports
        from scripts_v3.utils import normalize_name
        
        logging.info(f"Creating Parquet file from TSV: {self.tsv_path} -> {self.parquet_path}")
        
        # Read TSV file
        df = pd.read_csv(self.tsv_path, sep='\t', dtype=str, na_values='\\N')
        logging.info(f"Loaded {len(df)} names from TSV file")
        
        # Normalize names using the same function as credits data
        df['normalizedName'] = df['primaryName'].apply(lambda x: normalize_name(x) if pd.notna(x) else '')
        
        # Keep only the columns we need
        df_out = df[['nconst', 'normalizedName']]
        
        # Save to Parquet
        df_out.to_parquet(self.parquet_path, index=False)
        logging.info(f"Successfully created Parquet file: {self.parquet_path}")


def validate_credits_batch(episode_id: str, max_invalid_names: int = 50) -> List[Dict[str, Any]]:
    """
    Validate all names in credits for an episode and return problematic ones.
    
    Args:
        episode_id: Episode to validate
        max_invalid_names: Maximum number of invalid names to return
        
    Returns:
        List of problematic credits with validation info
    """
    import sqlite3
    
    validator = IMDBNameValidator()
    problematic_credits = []
    
    try:
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        
        # Get all credits for the episode
        cursor.execute(f"""
            SELECT id, role_group_normalized, name, role_detail, is_person,
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
            credit_id, role_group, name, role_detail, is_person, source_frame, frame_numbers, scene_pos, reviewed_status = credit
            
            # Skip if we've already processed this name
            if name in processed_names:
                continue
            
            processed_names.add(name)
            
            # Validate the name
            validation_result = validator.validate_name(name, role_group, is_person)
            
            # If name is not valid or has low confidence, add to problematic list
            if not validation_result['is_valid'] or validation_result['confidence'] < 0.8:
                problematic_credit = {
                    'id': credit_id,
                    'episode_id': episode_id,
                    'role_group': role_group,
                    'name': name,
                    'role_detail': role_detail,
                    'is_person': is_person,
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


def initialize_imdb_database() -> bool:
    """
    Initialize IMDB database if it doesn't exist.
    
    Returns:
        True if database exists or was created successfully
    """
    validator = IMDBNameValidator()
    return validator._initialize_imdb_data()
