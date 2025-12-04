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
from thefuzz import fuzz

from scripts_v3 import config
from scripts_v3.utils import generate_next_internal_code


# Global cache for IMDB data to support CLI usage without Streamlit session state
_IMDB_CACHE = {
    "initialized": False,
    "df": None,
    "name_lookup": None,
    "exact_index": {},
    "token_index": {}
}

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
    corrected_name: Optional[str] = None  # IMDB official name when auto-assigned


class IMDBNameValidator:
    """Handles IMDB name validation and nconst assignment using pandas DataFrame for performance"""
    
    def __init__(self, fuzzy_enabled: bool = True, fuzzy_threshold: int = 90):
        self.parquet_path = config.IMDB_PARQUET_PATH
        self.tsv_path = config.IMDB_TSV_PATH  # Fallback for backward compatibility
        self.fuzzy_enabled = fuzzy_enabled
        self.fuzzy_threshold = max(70, min(100, fuzzy_threshold))  # Clamp between 70-100
        self._imdb_df = None
        self._name_lookup = None
        self._initialized = False
        self._profession_filter_cache = {}  # Cache filtered DataFrames by profession set
        
        # NEW: Indici per velocizzare exact match e fuzzy search
        self._exact_index = {}  # normalizedName -> [row_indices]
        self._token_index = {}  # token -> {row_indices}
        
        # NEW: Cache validation results per (normalized_name, role_group)
        self._validation_cache = {}
        
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
        
        # Determine if we can use Streamlit session state
        use_streamlit_cache = False
        try:
            # Check if running within Streamlit context
            if getattr(st, "session_state", None) is not None and hasattr(st, "runtime") and st.runtime.exists():
                use_streamlit_cache = True
        except Exception:
            pass
            
        if use_streamlit_cache:
            if st.session_state.get(cache_key, False):
                self._initialized = True
                self._imdb_df = st.session_state.get("imdb_dataframe")
                self._name_lookup = st.session_state.get("imdb_name_lookup")
                self._exact_index = st.session_state.get("imdb_exact_index", {})
                self._token_index = st.session_state.get("imdb_token_index", {})
                if self._imdb_df is not None and self._name_lookup is not None:
                    logging.info("IMDB database loaded from Streamlit cache")
                    return True
        else:
            # Use global module-level cache for CLI/batch mode
            if _IMDB_CACHE["initialized"]:
                self._initialized = True
                self._imdb_df = _IMDB_CACHE["df"]
                self._name_lookup = _IMDB_CACHE["name_lookup"]
                self._exact_index = _IMDB_CACHE["exact_index"]
                self._token_index = _IMDB_CACHE["token_index"]
                if self._imdb_df is not None and self._name_lookup is not None:
                    logging.info("IMDB database loaded from global module cache")
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
                    na_values=['\\N'],
                    encoding='utf-8'
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
            
            # Ensure RangeIndex for fast indexing
            self._imdb_df = self._imdb_df.reset_index(drop=True)
            self._name_lookup = self._imdb_df
            
            load_time = time.time() - start_time
            logging.info(f"Successfully loaded {len(self._imdb_df):,} IMDB names in {load_time:.2f} seconds")
            
            # Build indices for fast lookup
            self._build_indices()
            
            # Cache the data
            if use_streamlit_cache:
                st.session_state[cache_key] = True
                st.session_state["imdb_dataframe"] = self._imdb_df
                st.session_state["imdb_name_lookup"] = self._name_lookup
                st.session_state["imdb_exact_index"] = self._exact_index
                st.session_state["imdb_token_index"] = self._token_index
            else:
                _IMDB_CACHE["initialized"] = True
                _IMDB_CACHE["df"] = self._imdb_df
                _IMDB_CACHE["name_lookup"] = self._name_lookup
                _IMDB_CACHE["exact_index"] = self._exact_index
                _IMDB_CACHE["token_index"] = self._token_index
            
            self._initialized = True
            return True
            
        except Exception as e:
            logging.error(f"Error loading IMDB data: {e}", exc_info=True)
            return False
    
    def _build_indices(self):
        """
        Build indices for fast exact match and fuzzy candidate selection:
        - _exact_index: normalizedName -> [row_indices]
        - _token_index: token -> {row_indices}
        """
        if self._name_lookup is None:
            return
        
        logging.info("🔧 Building indices for exact and fuzzy matching...")
        from collections import defaultdict
        
        self._exact_index = defaultdict(list)
        self._token_index = defaultdict(set)
        
        for idx, norm_name in self._name_lookup['normalized_name'].items():
            if pd.isna(norm_name):
                continue
            norm_name = str(norm_name).strip()
            if not norm_name:
                continue
            
            # Exact index
            self._exact_index[norm_name].append(idx)
            
            # Token index
            tokens = norm_name.split()
            for tok in tokens:
                if tok:
                    self._token_index[tok].add(idx)
        
        # Convert defaultdict to regular dict for session state storage
        self._exact_index = dict(self._exact_index)
        self._token_index = dict(self._token_index)
        
        logging.info(
            f"✅ Indices built: {len(self._exact_index)} unique normalized names, "
            f"{len(self._token_index)} unique tokens"
        )
    
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
        
        # OPTIMIZATION: Cache validation results by (normalized_name, role_group)
        from scripts_v3.utils import normalize_name
        norm_name = normalize_name(name)
        role_key = role_group.lower() if isinstance(role_group, str) else None
        cache_key = (norm_name, role_key, is_person)
        
        if cache_key in self._validation_cache:
            logging.debug(f"[VALIDATION CACHE HIT] Reusing result for '{name}' + '{role_group}'")
            return self._validation_cache[cache_key]
        
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
        
        # Check if role group is "Thanks" or "Unknown" - always gets internal codes (no IMDB search)
        if role_group and role_group.lower() in ['thanks', 'unknown']:
            logging.info(f"Processing '{name}' (role: '{role_group}' - always internal, no IMDB search)")
            
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
            internal_code = generate_next_internal_code(is_company=False)
            return ValidationResult(
                assigned_code=internal_code,
                assignment_status=CodeAssignmentStatus.INTERNAL_ASSIGNED,
                is_valid=True,
                confidence=1.0,
                matches=[],
                validation_method='thanks_internal_code',
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
            
            # Extract Jr./Sr. suffix BEFORE normalization (look for ", Jr." or ", Sr." pattern)
            original_name = name
            suffix = None
            import re
            suffix_match = re.search(r',\s*(Jr\.?|Sr\.?)\s*$', original_name, re.IGNORECASE)
            if suffix_match:
                suffix = suffix_match.group(1).lower().rstrip('.')
                # Remove the suffix from name before normalization
                name_for_normalization = original_name[:suffix_match.start()].strip()
                logging.debug(f"Extracted suffix '{suffix}' from original name '{original_name}', normalizing '{name_for_normalization}'")
            else:
                name_for_normalization = original_name
            
            # Normalize the name using the same function used for IMDB data
            normalized_name = normalize_name(name_for_normalization)
            
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
                base_permutations = [' '.join(words)]
            else:
                base_permutations = set([' '.join(p) for p in itertools.permutations(words)])
            
            # If we extracted a suffix (Jr./Sr.), append it to all permutations
            if suffix:
                permutations = set([f"{perm} {suffix}" for perm in base_permutations])
            else:
                permutations = base_permutations
            
            logging.info(f"IMDB validation for '{original_name}' -> normalized: '{normalized_name}' -> words: {words}{' + suffix: ' + suffix if suffix else ''} -> {len(permutations)} permutations")
            
            # First try exact matches for all permutations using index
            exact_matches = []
            is_fuzzy_match = False
            for perm in permutations:
                logging.debug(f"Searching IMDB for exact match: '{perm}'")
                
                try:
                    # OPTIMIZED: Use index for exact match
                    matches = self._exact_match_via_index(perm)
                    if matches:
                        logging.info(f"Found {len(matches)} IMDB exact match(es) for permutation '{perm}'")
                        exact_matches.extend(matches)
                    else:
                        logging.debug(f"No exact IMDB matches found for permutation '{perm}'")
                except Exception as e:
                    logging.error(f"Error searching IMDB for permutation '{perm}': {e}")
                    continue
            
            # If we found exact matches, use them
            if exact_matches:
                logging.info(f"Found {len(exact_matches)} exact match(es) for '{original_name}'")
                found_matches = exact_matches
                is_fuzzy_match = False
            else:
                # No exact matches - try fuzzy matching if enabled and threshold < 100
                # SKIP fuzzy matching for Thanks, Unknown and Additional Crew
                if role_group and role_group.lower() in ['thanks', 'unknown', 'additional crew']:
                    logging.info(f"Skipping fuzzy matching for role_group '{role_group}' - exact match only, assigning internal code")
                    internal_code = generate_next_internal_code(is_company=False)
                    return ValidationResult(
                        assigned_code=internal_code,
                        assignment_status=CodeAssignmentStatus.INTERNAL_ASSIGNED,
                        is_valid=False,
                        confidence=0.0,
                        matches=[],
                        validation_method='no_match_exact_only',
                        suggestion=None,
                        imdb_matches_json=None
                    )
                elif self.fuzzy_enabled and self.fuzzy_threshold < 100:
                    logging.info(f"No exact IMDB matches for '{original_name}', attempting fuzzy matching with name+profession (threshold: {self.fuzzy_threshold}%)")
                    fuzzy_matches = self._fuzzy_search_imdb(normalized_name, role_group=role_group, threshold=self.fuzzy_threshold)
                    
                    if fuzzy_matches:
                        logging.info(f"Found {len(fuzzy_matches)} fuzzy match(es) for '{original_name}' with similarity >= {self.fuzzy_threshold}%")
                        found_matches = fuzzy_matches
                        is_fuzzy_match = True
                    else:
                        logging.info(f"No fuzzy matches found for '{original_name}' at {self.fuzzy_threshold}% threshold, assigning internal code")
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
                else:
                    # Fuzzy matching disabled or threshold is 100%
                    logging.info(f"No exact IMDB matches for '{original_name}' and fuzzy matching is disabled (enabled: {self.fuzzy_enabled}, threshold: {self.fuzzy_threshold}%), assigning internal code")
                    internal_code = generate_next_internal_code(is_company=False)
                    return ValidationResult(
                        assigned_code=internal_code,
                        assignment_status=CodeAssignmentStatus.INTERNAL_ASSIGNED,
                        is_valid=False,
                        confidence=0.0,
                        matches=[],
                        validation_method='no_match_fuzzy_disabled',
                        suggestion=None,
                        imdb_matches_json=None
                    )
            
            # Now apply the assignment logic based on profession matching
            result = self._apply_assignment_logic(original_name, role_group, found_matches, is_fuzzy_match)
            
            # OPTIMIZATION: Cache the result
            self._validation_cache[cache_key] = result
            
            return result
            
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
    
    def _apply_assignment_logic(self, name: str, role_group: Optional[str], imdb_matches: List[Dict[str, Any]], is_fuzzy: bool = False) -> ValidationResult:
        """
        Apply the assignment logic based on profession matching.
        
        Args:
            name: Original name
            role_group: Role group for profession matching
            imdb_matches: List of IMDB matches found
            is_fuzzy: Whether matches came from fuzzy search
            
        Returns:
            ValidationResult with assigned code and status
        """
        logging.info(f"Applying assignment logic for '{name}' with role_group '{role_group}' and {len(imdb_matches)} IMDB matches (fuzzy: {is_fuzzy})")
        
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
            primary_name = match.get('primaryName')
            method = 'single_fuzzy_match' if is_fuzzy else 'single_exact_match'
            logging.info(f"Single exact match with compatible profession: {nconst} ({primary_name})")
            return ValidationResult(
                assigned_code=nconst,
                assignment_status=CodeAssignmentStatus.AUTO_ASSIGNED,
                is_valid=True,
                confidence=1.0,
                matches=compatible_matches,
                validation_method=method,
                suggestion=primary_name,
                imdb_matches_json=None,
                corrected_name=primary_name
            )
        
        elif total_matches > 1 and compatible_count == 1:
            # Rule 2: Multiple matches, but only one with compatible profession
            match = compatible_matches[0]
            nconst = match.get('nconst')
            primary_name = match.get('primaryName')
            method = 'multiple_fuzzy_one_compatible' if is_fuzzy else 'multiple_matches_one_compatible'
            logging.info(f"Multiple matches but only one compatible: {nconst} ({primary_name})")
            return ValidationResult(
                assigned_code=nconst,
                assignment_status=CodeAssignmentStatus.AUTO_ASSIGNED,
                is_valid=True,
                confidence=1.0,
                matches=compatible_matches,
                validation_method=method,
                suggestion=primary_name,
                imdb_matches_json=None,
                corrected_name=primary_name
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
    
    def _exact_match_via_index(self, normalized_name: str) -> List[Dict[str, Any]]:
        """Fast exact match using pre-built index"""
        if not normalized_name or not self._exact_index:
            return []
        
        indices = self._exact_index.get(normalized_name, [])
        if not indices:
            return []
        
        matches = self._name_lookup.iloc[indices]
        return matches.to_dict('records')
    
    def _get_candidate_indices_from_tokens(self, normalized_name: str) -> List[int]:
        """Get candidate row indices based on token overlap"""
        if not normalized_name or not self._token_index:
            return []
        
        tokens = str(normalized_name).split()
        if not tokens:
            return []
        
        # Union of all token candidate sets
        candidates = set()
        for tok in tokens:
            candidates |= self._token_index.get(tok, set())
        
        return list(candidates)
    
    def _fuzzy_search_imdb(self, normalized_name: str, role_group: Optional[str] = None, threshold: int = 90, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Perform fuzzy string matching against IMDB database using name+profession combinations.
        
        Args:
            normalized_name: Normalized name to search for
            role_group: Role group to map to IMDB professions for combined matching
            threshold: Minimum similarity score (0-100), default 90%
            max_results: Maximum number of results to return
            
        Returns:
            List of IMDB match dictionaries with similarity scores
        """
        if not normalized_name or self._name_lookup is None:
            return []
        
        try:
            # OPTIMIZATION 1: Get token-based candidates first (drastically reduces search space)
            candidate_indices = self._get_candidate_indices_from_tokens(normalized_name)
            if not candidate_indices:
                logging.info(f"Fuzzy search: no token-based candidates for '{normalized_name}'")
                return []
            
            # Get IMDB professions for this role group
            search_professions = []
            if role_group and config.has_imdb_profession_mapping(role_group):
                search_professions = list(config.get_imdb_professions_for_role_group(role_group))
            
            # OPTIMIZATION 2: PRE-FILTER by profession WITH CACHING!
            # Create cache key from sorted professions
            cache_key = tuple(sorted(search_professions)) if search_professions else None
            
            # Check if primaryProfession column exists in the DataFrame
            has_profession_column = 'primaryProfession' in self._name_lookup.columns
            
            if cache_key and cache_key in self._profession_filter_cache:
                # CACHE HIT - reuse filtered DataFrame!
                base_df = self._profession_filter_cache[cache_key]
                logging.debug(f"[CACHE HIT] Reusing profession filter for {search_professions} ({len(base_df):,} records)")
            elif search_professions and has_profession_column:
                # CACHE MISS - filter and cache result
                base_df = self._name_lookup
                profession_conditions = []
                for prof in search_professions:
                    # Check if profession appears in primaryProfession field
                    profession_conditions.append(
                        base_df['primaryProfession'].str.contains(prof, case=False, na=False, regex=False)
                    )
                
                if profession_conditions:
                    # Combine all profession conditions with OR
                    combined_condition = profession_conditions[0]
                    for condition in profession_conditions[1:]:
                        combined_condition |= condition
                    
                    base_df = base_df[combined_condition]
                    # Cache the filtered result
                    self._profession_filter_cache[cache_key] = base_df
                    logging.info(f"[FILTER] Profession filtering: {len(self._name_lookup):,} → {len(base_df):,} records ({search_professions}) - CACHED")
            else:
                # No profession filter (either no professions to filter, or column doesn't exist)
                base_df = self._name_lookup
                if search_professions and not has_profession_column:
                    logging.warning(f"primaryProfession column not found in IMDB data - skipping profession filtering. Consider regenerating the parquet file.")
                else:
                    logging.debug(f"No profession filter - using token candidates from ALL {len(base_df):,} records)")
            
            # OPTIMIZATION 3: Intersect profession filter with token candidates
            candidate_index = pd.Index(candidate_indices)
            search_df = base_df.loc[base_df.index.intersection(candidate_index)]
            
            if search_df.empty:
                logging.info(f"Fuzzy search: no candidates left after profession+token filtering for '{normalized_name}'")
                return []
            
            logging.info(f"Fuzzy search: {len(search_df):,} candidates (from {len(candidate_indices):,} token matches + profession filter)")
            
            # Create search strings with name+profession combinations
            search_strings = [normalized_name]  # Always include bare name as fallback
            
            if search_professions:
                for prof in search_professions:
                    search_strings.append(f"{normalized_name} {prof}")
                    # Special handling for cast - try both actor and actress
                    if prof in ['actor', 'actress']:
                        search_strings.append(f"{normalized_name} actor")
                        search_strings.append(f"{normalized_name} actress")
                
                # Remove duplicates
                search_strings = list(set(search_strings))
                logging.debug(f"Fuzzy search strings for '{normalized_name}' with role '{role_group}': {search_strings}")
            
            # Calculate fuzzy similarity scores
            matches_with_scores = []
            
            logging.info(f"Starting fuzzy matching on {len(search_df):,} candidates...")
            processed = 0
            
            for _, row in search_df.iterrows():
                processed += 1
                if processed % 10000 == 0:
                    logging.info(f"  Fuzzy matching progress: {processed:,}/{len(search_df):,} records...")
                
                # Get all search combinations for this IMDB entry
                imdb_search_combos = row.get('search_combinations', [])
                if not isinstance(imdb_search_combos, list):
                    # Fallback to just normalized name if search_combinations not available
                    imdb_search_combos = [row.get('normalized_name', '')]
                
                # Try matching against all combinations
                best_similarity = 0
                best_match_string = None
                
                for query_str in search_strings:
                    for imdb_str in imdb_search_combos:
                        if not imdb_str:
                            continue
                        
                        # MICRO-FILTER: Skip if length difference is too large
                        if abs(len(imdb_str) - len(query_str)) > 10:
                            continue
                        
                        # Use token_sort_ratio for better name matching (handles word order)
                        similarity = fuzz.token_sort_ratio(query_str, imdb_str)
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match_string = imdb_str
                
                if best_similarity >= threshold:
                    match_dict = row.to_dict()
                    match_dict['fuzzy_similarity'] = best_similarity
                    match_dict['matched_string'] = best_match_string
                    matches_with_scores.append(match_dict)
            
            # Sort by similarity score (descending) and limit results
            matches_with_scores.sort(key=lambda x: x['fuzzy_similarity'], reverse=True)
            top_matches = matches_with_scores[:max_results]
            
            if top_matches:
                logging.info(f"Fuzzy matching found {len(top_matches)} matches for '{normalized_name}' + role '{role_group}' (best: {top_matches[0]['fuzzy_similarity']}% via '{top_matches[0].get('matched_string', 'N/A')}')")
            
            return top_matches
            
        except Exception as e:
            logging.error(f"Error during fuzzy matching for '{normalized_name}': {e}", exc_info=True)
            return []
    
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
                # Skip search_combinations and other list/array columns
                if key == 'search_combinations' or isinstance(value, (list, tuple)):
                    continue  # Don't include list columns in formatted output
                
                # Handle scalar values
                try:
                    if pd.isna(value):
                        formatted_match[key] = None
                    elif key in ['birthYear', 'deathYear']:
                        try:
                            formatted_match[key] = int(value) if value else None
                        except (ValueError, TypeError):
                            formatted_match[key] = None
                    else:
                        formatted_match[key] = str(value) if value else ''
                except (ValueError, TypeError):
                    # If pd.isna() fails (e.g., on arrays), skip this field
                    continue
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
            
            # Count by profession categories (only if column exists)
            profession_stats = {}
            if 'primaryProfession' in self._imdb_df.columns:
                common_professions = ['actor', 'actress', 'director', 'writer', 'producer', 'composer']
                
                for profession in common_professions:
                    count = self._imdb_df['primaryProfession'].fillna('').str.contains(profession, case=False).sum()
                    profession_stats[profession] = int(count)
            else:
                profession_stats = {'note': 'primaryProfession column not available in parquet file'}
            
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
        Includes search_combinations for enhanced fuzzy matching.
        """
        import pandas as pd
        # Import normalize_name locally to avoid circular imports
        from scripts_v3.utils import normalize_name
        
        logging.info(f"Creating Parquet file from TSV: {self.tsv_path} -> {self.parquet_path}")
        
        # Read TSV file
        df = pd.read_csv(self.tsv_path, sep='\t', dtype=str, na_values='\\N', encoding='utf-8')
        logging.info(f"Loaded {len(df)} names from TSV file")
        
        # Normalize names using the same function as credits data
        df['normalizedName'] = df['primaryName'].apply(lambda x: normalize_name(x) if pd.notna(x) else '')
        
        # Create search_combinations for enhanced fuzzy matching (name + profession combos)
        def create_search_combinations(row):
            """Create all possible name+profession combinations for fuzzy matching."""
            normalized_name = row['normalizedName']
            if pd.isna(normalized_name) or not normalized_name:
                return []
            
            professions = row.get('primaryProfession', '')
            if pd.isna(professions) or not professions:
                # No professions listed - just use name
                return [normalized_name]
            
            # Split professions and create combinations
            prof_list = [p.strip() for p in str(professions).split(',') if p.strip()]
            combinations = []
            
            for prof in prof_list:
                # Add name + profession combination
                combinations.append(f"{normalized_name} {prof}")
            
            # Remove duplicates while preserving order
            seen = set()
            unique_combinations = []
            for combo in combinations:
                if combo not in seen:
                    seen.add(combo)
                    unique_combinations.append(combo)
            
            return unique_combinations
        
        logging.info("Creating search combinations for enhanced fuzzy matching...")
        df['search_combinations'] = df.apply(create_search_combinations, axis=1)
        
        # Keep columns needed for matching
        columns_to_keep = ['nconst', 'normalizedName', 'primaryName', 'primaryProfession', 'birthYear', 'deathYear', 'search_combinations']
        # Only keep columns that exist in the DataFrame
        existing_columns = [col for col in columns_to_keep if col in df.columns]
        df_out = df[existing_columns]
        
        # Save to Parquet
        df_out.to_parquet(self.parquet_path, index=False)
        logging.info(f"Successfully created Parquet file with columns {existing_columns}: {self.parquet_path}")

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
