"""
IMDB Name Validation Module with nconst Assignment

This module handles:
1. Loading IMDB names database from name.basics.parquet using pandas DataFrame
2. Validating extracted names against IMDB database using normalized names
3. Assigning nconst codes based on role group and profession matching
4. Generating internal progressive codes (gp1234567) for non-IMDB people
5. Handling ambiguous cases for manual review

Assignment Rules:
- Single person with compatible profession → automatic nconst assignment
- Multiple people, one with compatible profession → automatic nconst assignment  
- Multiple people with compatible professions → manual review required
- Thanks/Additional crew → always internal gp code
- Not in IMDB → internal gp code
- In IMDB but incompatible profession → manual review with suggestion
- Multiple profiles, no compatible profession → manual review with internal default
"""

import itertools
import json
import logging
import math
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import streamlit as st

from scripts_v3 import config
from scripts_v3.utils import generate_next_internal_code


class CodeAssignmentStatus(Enum):
    """Enum for code assignment status values"""
    AUTO_ASSIGNED = "auto_assigned"
    MANUAL_REQUIRED = "manual_required"
    AMBIGUOUS = "ambiguous"
    INTERNAL_ASSIGNED = "internal_assigned"


@dataclass
class IMDBPerson:
    """Represents a person from IMDB name.basics.tsv"""
    nconst: str
    primary_name: str
    birth_year: Optional[int]
    death_year: Optional[int]
    primary_profession: List[str]
    known_for_titles: List[str]


@dataclass
class ValidationResult:
    """Result of IMDB name validation with code assignment"""
    assigned_code: str
    assignment_status: CodeAssignmentStatus
    is_valid: bool
    confidence: float
    matches: List[Dict[str, Any]]
    validation_method: str
    suggestion: Optional[str]
    imdb_matches_json: Optional[str]  # JSON string of matches for ambiguous cases


class IMDBNameValidator:
    """Handles IMDB name validation and nconst assignment using pandas DataFrame for performance"""
    
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
    
    def validate_name_with_code_assignment(self, name: str, role_group: Optional[str] = None, is_person: Optional[bool] = None) -> ValidationResult:
        """
        Validate a name against IMDB database and assign appropriate code (nconst or internal).
        
        Args:
            name: Name to validate
            role_group: Role group for profession matching
            is_person: Explicit flag if name refers to a person (True) or company (False)
            
        Returns:
            ValidationResult with assigned code and status
        """
        logging.debug(f"IMDB validation with code assignment for name: '{name}', role_group: '{role_group}', is_person: {is_person}")
        
        # Import the helper function for consistent company detection (fallback)
        from scripts_v3.utils import is_company_role_group
        
        # First check if we have explicit is_person information
        if is_person is False or is_person == 0:
            logging.info(f"Processing company name: '{name}' (is_person={is_person})")
            
            # First check if we already have an internal code for this company name and role
            from scripts_v3.utils import normalize_name
            normalized_name = normalize_name(name)
            
            if normalized_name and role_group:
                existing_code = self._check_existing_internal_code(normalized_name, role_group, is_company=True)
                if existing_code:
                    logging.info(f"Found existing internal code '{existing_code}' for company '{name}' in role '{role_group}'")
                    return ValidationResult(
                        assigned_code=existing_code,
                        assignment_status=CodeAssignmentStatus.INTERNAL_ASSIGNED,
                        is_valid=True,
                        confidence=1.0,
                        matches=[],
                        validation_method='reused_existing_internal_code',
                        suggestion=None,
                        imdb_matches_json=None
                    )
            
            # If no existing code found, create a new one
            internal_code = generate_next_internal_code(is_company=True)
            return ValidationResult(
                assigned_code=internal_code,
                assignment_status=CodeAssignmentStatus.INTERNAL_ASSIGNED,
                is_valid=True,
                confidence=1.0,
                matches=[],
                validation_method='company_name_internal_code',
                suggestion=None,
                imdb_matches_json=None
            )
        
        # If is_person is explicitly True (1), treat as person regardless of role group
        elif is_person is True or is_person == 1:
            logging.debug(f"Explicit is_person=1 for '{name}', treating as person regardless of role group '{role_group}'")
            # Continue with person validation logic below
            pass
        
        # Fallback to role group detection only when is_person is None/unclear
        elif is_person is None and is_company_role_group(role_group):
            logging.info(f"Processing company name: '{name}' (role: '{role_group}', is_person unspecified)")
            
            # First check if we already have an internal code for this company name and role
            from scripts_v3.utils import normalize_name
            normalized_name = normalize_name(name)
            
            if normalized_name:
                existing_code = self._check_existing_internal_code(normalized_name, role_group, is_company=True)
                if existing_code:
                    logging.info(f"Found existing internal code '{existing_code}' for company '{name}' in role '{role_group}'")
                    return ValidationResult(
                        assigned_code=existing_code,
                        assignment_status=CodeAssignmentStatus.INTERNAL_ASSIGNED,
                        is_valid=True,
                        confidence=1.0,
                        matches=[],
                        validation_method='reused_existing_internal_code',
                        suggestion=None,
                        imdb_matches_json=None
                    )
            
            # If no existing code found, create a new one
            internal_code = generate_next_internal_code(is_company=True)
            return ValidationResult(
                assigned_code=internal_code,
                assignment_status=CodeAssignmentStatus.INTERNAL_ASSIGNED,
                is_valid=True,
                confidence=1.0,
                matches=[],
                validation_method='company_name_internal_code_role_group',
                suggestion=None,
                imdb_matches_json=None
            )
        
        # Check if role group always gets internal codes (thanks, additional crew)
        if role_group and role_group.lower() in ['thanks', 'additional crew']:
            logging.info(f"Processing '{name}' (role: '{role_group}' - always internal)")
            
            # First check if we already have an internal code for this name and role
            from scripts_v3.utils import normalize_name
            normalized_name = normalize_name(name)
            
            if normalized_name:
                existing_code = self._check_existing_internal_code(normalized_name, role_group, is_company=False)
                if existing_code:
                    logging.info(f"Found existing internal code '{existing_code}' for '{name}' in role '{role_group}'")
                    return ValidationResult(
                        assigned_code=existing_code,
                        assignment_status=CodeAssignmentStatus.INTERNAL_ASSIGNED,
                        is_valid=True,
                        confidence=1.0,
                        matches=[],
                        validation_method='reused_existing_internal_code',
                        suggestion=None,
                        imdb_matches_json=None
                    )
            
            # If no existing code found, create a new one
            internal_code = generate_next_internal_code(is_company=False)  # Person code for thanks/additional crew
            return ValidationResult(
                assigned_code=internal_code,
                assignment_status=CodeAssignmentStatus.INTERNAL_ASSIGNED,
                is_valid=True,
                confidence=1.0,
                matches=[],
                validation_method='thanks_additional_crew_internal_code',
                suggestion=None,
                imdb_matches_json=None
            )
        
        if not name or len(name.strip()) < 2:
            logging.warning(f"Name '{name}' too short, assigning internal code")
            internal_code = generate_next_internal_code(is_company=False)
            return ValidationResult(
                assigned_code=internal_code,
                assignment_status=CodeAssignmentStatus.INTERNAL_ASSIGNED,
                is_valid=False,
                confidence=0.0,
                matches=[],
                validation_method='name_too_short',
                suggestion=None,
                imdb_matches_json=None
            )
        
        # Initialize IMDB data if needed
        if not self._initialize_imdb_data():
            logging.error(f"IMDB database initialization failed for name '{name}', assigning internal code")
            internal_code = generate_next_internal_code(is_company=False)
            return ValidationResult(
                assigned_code=internal_code,
                assignment_status=CodeAssignmentStatus.INTERNAL_ASSIGNED,
                is_valid=False,
                confidence=0.0,
                matches=[],
                validation_method='database_error',
                suggestion=None,
                imdb_matches_json=None
            )
        
        try:
            # Import normalize_name locally to avoid circular imports
            from scripts_v3.utils import normalize_name
            
            # Normalize the name using the same function used for IMDB data
            original_name = name
            normalized_name = normalize_name(name)
            
            if not normalized_name:
                logging.warning(f"Name '{original_name}' became empty after normalization, assigning internal code")
                internal_code = generate_next_internal_code(is_company=False)
                return ValidationResult(
                    assigned_code=internal_code,
                    assignment_status=CodeAssignmentStatus.INTERNAL_ASSIGNED,
                    is_valid=False,
                    confidence=0.0,
                    matches=[],
                    validation_method='invalid_name_after_normalization',
                    suggestion=None,
                    imdb_matches_json=None
                )
            
            # Check if we already have an internal code for this normalized name and role
            is_company = (is_person is False or is_person == 0)
            existing_code = self._check_existing_internal_code(normalized_name, role_group, is_company)
            if existing_code:
                logging.info(f"Found existing internal code '{existing_code}' for normalized name '{normalized_name}' in role '{role_group}'")
                return ValidationResult(
                    assigned_code=existing_code,
                    assignment_status=CodeAssignmentStatus.INTERNAL_ASSIGNED,
                    is_valid=True,
                    confidence=1.0,
                    matches=[],
                    validation_method='reused_existing_internal_code',
                    suggestion=None,
                    imdb_matches_json=None
                )
            
            # Split normalized name into words for permutation testing
            words = normalized_name.split()
            if len(words) == 0:
                logging.warning(f"No valid words found in normalized name '{normalized_name}', assigning internal code")
                internal_code = generate_next_internal_code(is_company=False)
                return ValidationResult(
                    assigned_code=internal_code,
                    assignment_status=CodeAssignmentStatus.INTERNAL_ASSIGNED,
                    is_valid=False,
                    confidence=0.0,
                    matches=[],
                    validation_method='no_valid_words',
                    suggestion=None,
                    imdb_matches_json=None
                )
            
            # Safeguard: detect likely concatenated names (too many words)
            if len(words) > 6:
                logging.warning(f"Name '{original_name}' has {len(words)} words - likely concatenated multiple names, assigning internal code")
                internal_code = generate_next_internal_code(is_company=False)
                return ValidationResult(
                    assigned_code=internal_code,
                    assignment_status=CodeAssignmentStatus.INTERNAL_ASSIGNED,
                    is_valid=False,
                    confidence=0.0,
                    matches=[],
                    validation_method='concatenated_names_detected',
                    suggestion='This appears to be multiple names concatenated together - needs manual review',
                    imdb_matches_json=None
                )
            
            # Safeguard: limit the number of words to prevent combinatorial explosion
            # Increased limit to allow more complex names
            if len(words) > 6:
                logging.warning(f"Name '{original_name}' has {len(words)} words - limiting to first 6 words for validation")
                words = words[:6]
            
            # Generate permutations with an increased limit
            max_permutations = 720  # 6! = 720, higher limit for better matching
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
            
            # If no matches found, assign internal code
            if not found_matches:
                logging.info(f"No IMDB matches found for '{original_name}', assigning internal code")
                internal_code = generate_next_internal_code(is_company=False)
                return ValidationResult(
                    assigned_code=internal_code,
                    assignment_status=CodeAssignmentStatus.INTERNAL_ASSIGNED,
                    is_valid=False,
                    confidence=0.0,
                    matches=[],
                    validation_method='no_match',
                    suggestion=None,
                    imdb_matches_json=None
                )
            
            # Now apply the assignment logic based on profession matching
            return self._apply_assignment_logic(original_name, role_group, found_matches)
            
        except Exception as e:
            logging.error(f"Error validating name '{name}': {e}", exc_info=True)
            internal_code = generate_next_internal_code(is_company=False)
            return ValidationResult(
                assigned_code=internal_code,
                assignment_status=CodeAssignmentStatus.INTERNAL_ASSIGNED,
                is_valid=False,
                confidence=0.0,
                matches=[],
                validation_method='error',
                suggestion=None,
                imdb_matches_json=None
            )
    
    def _apply_assignment_logic(self, name: str, role_group: Optional[str], imdb_matches: List[Dict[str, Any]]) -> ValidationResult:
        """
        Apply the assignment logic based on profession matching.
        
        Args:
            name: Original name
            role_group: Role group for profession matching
            imdb_matches: List of IMDB matches found
            
        Returns:
            ValidationResult with assigned code and status
        """
        logging.info(f"Applying assignment logic for '{name}' with role_group '{role_group}' and {len(imdb_matches)} IMDB matches")
        
        # Format the matches for consistency
        formatted_matches = self._format_matches(imdb_matches)
        
        # Check if role group has IMDB profession mapping
        if not role_group or not config.has_imdb_profession_mapping(role_group):
            logging.info(f"Role group '{role_group}' has no IMDB profession mapping, assigning internal code")
            internal_code = generate_next_internal_code(is_company=False)
            return ValidationResult(
                assigned_code=internal_code,
                assignment_status=CodeAssignmentStatus.INTERNAL_ASSIGNED,
                is_valid=True,
                confidence=1.0,
                matches=formatted_matches,
                validation_method='no_profession_mapping',
                suggestion=None,
                imdb_matches_json=None
            )
        
        # Get expected IMDB professions for this role group
        expected_professions = config.get_imdb_professions_for_role_group(role_group)
        logging.info(f"Expected professions for role '{role_group}': {expected_professions}")
        
        # Filter matches by profession compatibility
        compatible_matches = []
        for match in formatted_matches:
            match_professions = self._extract_professions_from_match(match)
            if self._has_profession_overlap(match_professions, expected_professions):
                compatible_matches.append(match)
        
        logging.info(f"Found {len(compatible_matches)} compatible matches out of {len(formatted_matches)} total matches")
        
        # Apply assignment rules based on number of matches
        total_matches = len(formatted_matches)
        compatible_count = len(compatible_matches)
        
        if total_matches == 1 and compatible_count == 1:
            # Rule 1: Single exact match with compatible profession
            match = compatible_matches[0]
            nconst = match.get('nconst')
            logging.info(f"Single exact match with compatible profession: {nconst}")
            return ValidationResult(
                assigned_code=nconst,
                assignment_status=CodeAssignmentStatus.AUTO_ASSIGNED,
                is_valid=True,
                confidence=1.0,
                matches=compatible_matches,
                validation_method='single_exact_match',
                suggestion=match.get('primaryName'),
                imdb_matches_json=None
            )
        
        elif total_matches > 1 and compatible_count == 1:
            # Rule 2: Multiple matches, but only one with compatible profession
            match = compatible_matches[0]
            nconst = match.get('nconst')
            logging.info(f"Multiple matches but only one compatible: {nconst}")
            return ValidationResult(
                assigned_code=nconst,
                assignment_status=CodeAssignmentStatus.AUTO_ASSIGNED,
                is_valid=True,
                confidence=1.0,
                matches=compatible_matches,
                validation_method='multiple_matches_one_compatible',
                suggestion=match.get('primaryName'),
                imdb_matches_json=None
            )
        
        elif compatible_count > 1:
            # Rule 3: Multiple matches with compatible professions - ambiguous
            logging.info(f"Ambiguous case: {compatible_count} matches with compatible professions")
            return ValidationResult(
                assigned_code=None,  # No code assigned yet
                assignment_status=CodeAssignmentStatus.AMBIGUOUS,
                is_valid=False,
                confidence=0.5,
                matches=compatible_matches,
                validation_method='multiple_compatible_matches',
                suggestion=None,
                imdb_matches_json=json.dumps(compatible_matches)
            )
        
        elif total_matches > 0 and compatible_count == 0:
            # Rule 6: In IMDB but no compatible profession
            logging.info(f"Found in IMDB but no compatible profession")
            best_match = self._select_best_match(formatted_matches, role_group)
            return ValidationResult(
                assigned_code=None,  # No code assigned yet
                assignment_status=CodeAssignmentStatus.MANUAL_REQUIRED,
                is_valid=False,
                confidence=0.3,
                matches=formatted_matches,
                validation_method='incompatible_profession',
                suggestion=best_match.get('nconst') if best_match else None,
                imdb_matches_json=json.dumps(formatted_matches)
            )
        
        else:
            # Fallback case - assign internal code
            logging.info(f"Fallback case: assigning internal code")
            internal_code = generate_next_internal_code(is_company=False)
            return ValidationResult(
                assigned_code=internal_code,
                assignment_status=CodeAssignmentStatus.INTERNAL_ASSIGNED,
                is_valid=False,
                confidence=0.0,
                matches=formatted_matches,
                validation_method='fallback_internal',
                suggestion=None,
                imdb_matches_json=None
            )
    
    def _extract_professions_from_match(self, match: Dict[str, Any]) -> Set[str]:
        """
        Extract profession set from an IMDB match.
        
        Args:
            match: IMDB match dictionary
            
        Returns:
            Set of professions (lowercase)
        """
        professions_str = match.get('primaryProfession', '')
        if not professions_str:
            return set()
        
        # Split by comma and normalize
        professions = {prof.strip().lower() for prof in professions_str.split(',') if prof.strip()}
        return professions
    
    def _has_profession_overlap(self, match_professions: Set[str], expected_professions: Set[str]) -> bool:
        """
        Check if there's any overlap between match professions and expected professions.
        
        Args:
            match_professions: Set of professions from IMDB match
            expected_professions: Set of expected professions for role group
            
        Returns:
            bool: True if there's any overlap
        """
        return len(match_professions.intersection(expected_professions)) > 0
    
    def validate_name(self, name: str, role_group: Optional[str] = None, is_person: Optional[bool] = None) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility. 
        Now calls the new method and converts the result to the old format.
        
        Args:
            name: Name to validate
            role_group: Role group for context
            is_person: Explicit flag if name refers to a person
            
        Returns:
            Dictionary with validation results (legacy format)
        """
        result = self.validate_name_with_code_assignment(name, role_group, is_person)
        
        # Convert to legacy format
        return {
            'is_valid': result.is_valid,
            'confidence': result.confidence,
            'matches': result.matches,
            'validation_method': result.validation_method,
            'suggestion': result.suggestion
        }
    
    def validate_name_with_normalized(self, name: str, normalized_name: Optional[str] = None, 
                                     role_group: Optional[str] = None, is_person: Optional[bool] = None) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility with pre-computed normalized names.
        Now calls the new method and converts the result to the old format.
        
        Args:
            name: Original name to validate
            normalized_name: Pre-computed normalized name (ignored in new implementation)
            role_group: Role group for context
            is_person: Explicit flag if name refers to a person
            
        Returns:
            Dictionary with validation results (legacy format)
        """
        result = self.validate_name_with_code_assignment(name, role_group, is_person)
        
        # Convert to legacy format
        return {
            'is_valid': result.is_valid,
            'confidence': result.confidence,
            'matches': result.matches,
            'validation_method': result.validation_method,
            'suggestion': result.suggestion
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

    def _check_existing_internal_code(self, normalized_name: str, role_group: str, is_company: bool) -> Optional[str]:
        """
        Check if we already have an internal code (gp/cm) for this normalized name and role.
        
        Args:
            normalized_name: Normalized name to search for
            role_group: Role group to match
            is_company: Whether this is a company (True) or person (False)
            
        Returns:
            Existing internal code if found, None otherwise
        """
        try:
            import sqlite3
            
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            # Determine the code prefix to search for
            code_prefix = 'cm' if is_company else 'gp'
            
            # Search for existing credits with the same normalized name, role group, and internal code type
            # We use the normalize_name function to ensure consistency
            from scripts_v3.utils import normalize_name
            
            cursor.execute(f"""
                SELECT assigned_code, name, role_group_normalized
                FROM {config.DB_TABLE_CREDITS} 
                WHERE assigned_code LIKE ? 
                AND code_assignment_status = 'internal_assigned'
                AND role_group_normalized = ?
                ORDER BY assigned_code ASC
            """, (f"{code_prefix}%", role_group))
            
            results = cursor.fetchall()
            conn.close()
            
            # Check each result to see if the normalized name matches
            for assigned_code, stored_name, stored_role in results:
                if stored_name:
                    stored_normalized = normalize_name(stored_name)
                    if stored_normalized == normalized_name:
                        logging.info(f"Found existing internal code '{assigned_code}' for normalized name '{normalized_name}' in role '{role_group}' (original: '{stored_name}')")
                        return assigned_code
            
            logging.debug(f"No existing internal code found for normalized name '{normalized_name}' in role '{role_group}'")
            return None
            
        except Exception as e:
            logging.error(f"Error checking existing internal codes: {e}")
            return None


def validate_credits_batch_with_code_assignment(episode_id: str, max_problematic_names: int = 50) -> List[Dict[str, Any]]:
    """
    Validate all names in credits for an episode, assign codes, and return problematic ones.
    
    Args:
        episode_id: Episode to validate
        max_problematic_names: Maximum number of problematic names to return
        
    Returns:
        List of problematic credits that need manual review
    """
    import sqlite3
    
    validator = IMDBNameValidator()
    problematic_credits = []
    
    try:
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        
        # Get all credits for the episode that need code assignment
        cursor.execute(f"""
            SELECT id, role_group_normalized, name, role_detail, is_person,
                   source_frame, original_frame_number, scene_position, reviewed_status,
                   assigned_code, code_assignment_status
            FROM {config.DB_TABLE_CREDITS} 
            WHERE episode_id = ? AND (assigned_code IS NULL OR code_assignment_status IN ('manual_required', 'ambiguous'))
            ORDER BY role_group_normalized, name
        """, (episode_id,))
        
        credits_data = cursor.fetchall()
        
        if not credits_data:
            conn.close()
            return []
        
        logging.info(f"Processing {len(credits_data)} credits for code assignment in episode {episode_id}")
        
        processed_names = set()  # Avoid duplicate validation
        
        for credit in credits_data:
            credit_id, role_group, name, role_detail, is_person, source_frame, frame_numbers, scene_pos, reviewed_status, assigned_code, code_assignment_status = credit
            
            # Skip if we've already processed this name
            name_key = (name, role_group)
            if name_key in processed_names:
                continue
            
            processed_names.add(name_key)
            
            # Validate the name and assign code
            validation_result = validator.validate_name_with_code_assignment(name, role_group, is_person)
            
            # Update the database with the assigned code
            update_query = f"""
                UPDATE {config.DB_TABLE_CREDITS} 
                SET assigned_code = ?, code_assignment_status = ?, imdb_matches = ?
                WHERE id = ?
            """
            
            cursor.execute(update_query, (
                validation_result.assigned_code,
                validation_result.assignment_status.value,
                validation_result.imdb_matches_json,
                credit_id
            ))
            
            # If manual review is required, add to problematic list
            if validation_result.assignment_status in [CodeAssignmentStatus.MANUAL_REQUIRED, CodeAssignmentStatus.AMBIGUOUS]:
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
                    'problem_types': ['code_assignment_required'],
                    'priority_score': 90 if validation_result.assignment_status == CodeAssignmentStatus.AMBIGUOUS else 70
                }
                
                problematic_credits.append(problematic_credit)
                
                if len(problematic_credits) >= max_problematic_names:
                    break
        
        conn.commit()
        conn.close()
        
        # Sort by priority score (ambiguous cases first)
        problematic_credits.sort(key=lambda x: x['priority_score'], reverse=True)
        
        logging.info(f"Found {len(problematic_credits)} credits needing manual review for episode {episode_id}")
        return problematic_credits
        
    except Exception as e:
        logging.error(f"Error processing credits batch for episode {episode_id}: {e}", exc_info=True)
        return []


def validate_credits_batch(episode_id: str, max_invalid_names: int = 50) -> List[Dict[str, Any]]:
    """
    Legacy function for backward compatibility.
    Now calls the new code assignment function.
    
    Args:
        episode_id: Episode to validate
        max_invalid_names: Maximum number of invalid names to return
        
    Returns:
        List of problematic credits with validation info
    """
    return validate_credits_batch_with_code_assignment(episode_id, max_invalid_names)


def initialize_imdb_database() -> bool:
    """
    Initialize IMDB database if it doesn't exist.
    
    Returns:
        True if database exists or was created successfully
    """
    validator = IMDBNameValidator()
    return validator._initialize_imdb_data()
