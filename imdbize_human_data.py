#!/usr/bin/env python3
"""
Script to apply IMDB validation/correction to human-corrected credits CSV.
Standalone version that replicates the exact logic from imdb_name_validation.py
without Streamlit dependencies.

Adds a new column 'nome_corretto_imdb' with the canonical IMDB name when found.
"""

import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import logging
import re
from thefuzz import fuzz  # SAME as original - not fuzzywuzzy!
from typing import Optional, List, Dict, Any
from enum import Enum
import itertools
import math
from collections import defaultdict

# Add scripts_v3 to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts_v3 import config
from scripts_v3.utils import (
    normalize_name,
    normalize_name_with_nickname,
    strip_honorifics,
    strip_parentheticals,
    strip_quoted_asides,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class CodeAssignmentStatus(Enum):
    """Status of IMDB code assignment"""
    AUTO_ASSIGNED = "auto_assigned"
    AMBIGUOUS = "ambiguous"
    INTERNAL_ASSIGNED = "internal_assigned"


class StandaloneIMDBValidator:
    """
    Standalone IMDB validator without Streamlit dependencies.
    Replicates logic from IMDBNameValidator including ALL edge cases,
    ma con ottimizzazioni per dataset molto grandi.
    """

    def __init__(self, fuzzy_enabled: bool = True, fuzzy_threshold: int = 90):
        self.parquet_path = config.IMDB_PARQUET_PATH
        self.fuzzy_enabled = fuzzy_enabled
        self.fuzzy_threshold = max(70, min(100, fuzzy_threshold))
        self._name_lookup: Optional[pd.DataFrame] = None

        # Indici per velocizzare le ricerche
        self._exact_index: Dict[str, List[int]] = defaultdict(list)   # normalizedName -> [row_indices]
        self._token_index: Dict[str, set] = defaultdict(set)          # token -> {row_indices}
        self._exact_index_with_nickname: Dict[str, List[int]] = defaultdict(list)  # normalizedNameWithNickname -> [row_indices]

        # Cache per DataFrame filtrati per professione
        self._profession_filter_cache: Dict[Any, pd.DataFrame] = {}

        # Load IMDB data
        self._load_imdb_data()

    def _load_imdb_data(self):
        """Load IMDB data from parquet file e costruisci indici."""
        if not self.parquet_path.exists():
            logging.error(f"❌ IMDB parquet file not found: {self.parquet_path}")
            return

        try:
            print(f"📖 Loading IMDB database from parquet (this may take 30-60 seconds)...")
            logging.info(f"📖 Loading IMDB database from: {self.parquet_path}")
            df = pd.read_parquet(self.parquet_path)
            print(f"✅ IMDB database loaded successfully!")

            # Verify required columns (use ACTUAL column names from parquet!)
            required_cols = ['nconst', 'primaryName', 'normalizedName', 'primaryProfession']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                logging.error(f"❌ Parquet missing columns: {missing}")
                return

            # Assicuriamoci di avere una RangeIndex semplice (0..N-1)
            df = df.reset_index(drop=True)

            self._name_lookup = df
            logging.info(f"✅ Loaded {len(self._name_lookup)} IMDB records")

            # Costruisci indici per exact match e fuzzy candidate selection
            self._build_indices()

        except Exception as e:
            logging.error(f"❌ Error loading IMDB data: {e}", exc_info=True)
            self._name_lookup = None

    def _build_indices(self):
        """Costruisce:
        - _exact_index: normalizedName -> lista di row index
        - _token_index: token -> set di row index
        per accelerare exact e fuzzy matching.
        """
        if self._name_lookup is None:
            return

        logging.info("🔧 Building indices for exact and fuzzy matching...")
        for idx, norm_name in self._name_lookup['normalizedName'].items():
            if pd.isna(norm_name):
                continue
            norm_name = str(norm_name).strip()
            if not norm_name:
                continue

            # indice esatto
            self._exact_index[norm_name].append(idx)

            # indice per token
            tokens = norm_name.split()
            for tok in tokens:
                if tok:
                    self._token_index[tok].add(idx)

        if 'normalizedNameWithNickname' in self._name_lookup.columns:
            for idx, nick_name in self._name_lookup['normalizedNameWithNickname'].items():
                if pd.isna(nick_name):
                    continue
                nick_name = str(nick_name).strip()
                if not nick_name:
                    continue
                self._exact_index_with_nickname[nick_name].append(idx)
        else:
            logging.warning(
                "normalizedNameWithNickname column not found in parquet - "
                "nickname-first matching disabled (regenerate the parquet to enable it)"
            )

        logging.info(
            f"✅ Indices built: {len(self._exact_index)} unique normalizedName entries, "
            f"{len(self._token_index)} unique tokens, "
            f"{len(self._exact_index_with_nickname)} unique normalizedNameWithNickname entries"
        )

    def _exact_match_imdb(self, normalized_name: str) -> List[Dict[str, Any]]:
        """Exact string match against IMDB normalized names via indice."""
        if self._name_lookup is None or not normalized_name:
            return []

        indices = self._exact_index.get(normalized_name, [])
        if not indices:
            return []

        matches = self._name_lookup.iloc[indices]
        return matches.to_dict('records')

    def _exact_match_imdb_with_nickname(self, normalized_name_with_nickname: str) -> List[Dict[str, Any]]:
        """Same as _exact_match_imdb, but against the with-nickname index
        (nickname aside kept, canonicalized to "..."). Used as a first-pass
        lookup for names with a nickname aside, since it can be the only
        thing that disambiguates a common name among many IMDB namesakes."""
        if self._name_lookup is None or not normalized_name_with_nickname:
            return []

        indices = self._exact_index_with_nickname.get(normalized_name_with_nickname, [])
        if not indices:
            return []

        matches = self._name_lookup.iloc[indices]
        return matches.to_dict('records')

    def _candidate_indices_from_tokens(self, normalized_name: str) -> List[int]:
        """Restituisce una lista di indici candidati basata sui token del nome."""
        if not normalized_name:
            return []

        tokens = str(normalized_name).split()
        if not tokens:
            return []

        candidate_sets = [self._token_index.get(tok, set()) for tok in tokens]
        if not candidate_sets:
            return []

        # union di tutti i set di candidati
        candidates = set()
        for s in candidate_sets:
            candidates |= s

        return list(candidates)

    def _fuzzy_search_imdb(self, normalized_name: str, role_group: Optional[str] = None,
                           threshold: int = 90, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Fuzzy search con profession-based filtering,
        usando un inverted index sui token per ridurre il numero di candidati.
        """
        if self._name_lookup is None or not normalized_name:
            return []

        try:
            # 1) Candidati dai token (riduce drasticamente)
            candidate_indices = self._candidate_indices_from_tokens(normalized_name)
            if not candidate_indices:
                logging.info(
                    f"Fuzzy search: no token-based candidates for '{normalized_name}'"
                )
                return []

            # 2) Get profession mapping (come prima)
            search_professions: List[str] = []
            if role_group and config.has_imdb_profession_mapping(role_group):
                search_professions = list(
                    config.get_imdb_professions_for_role_group(role_group)
                )

            # 3) Profession filter + caching (come prima, ma poi si interseca con candidati)
            if search_professions:
                cache_key = tuple(sorted(search_professions))
                if cache_key in self._profession_filter_cache:
                    base_df = self._profession_filter_cache[cache_key]
                    print(
                        f"[CACHE HIT] Reusing profession filter for {search_professions} "
                        f"({len(base_df):,} records)"
                    )
                else:
                    base_df = self._name_lookup
                    profession_conditions = []
                    for prof in search_professions:
                        profession_conditions.append(
                            base_df['primaryProfession'].str.contains(
                                prof, case=False, na=False, regex=False
                            )
                        )

                    if profession_conditions:
                        combined_condition = profession_conditions[0]
                        for cond in profession_conditions[1:]:
                            combined_condition |= cond

                        base_df = base_df[combined_condition]
                        self._profession_filter_cache[cache_key] = base_df
                        print(
                            f"[FILTER] Profession: {len(self._name_lookup):,} → {len(base_df):,} "
                            f"records ({search_professions}) - CACHED"
                        )
                        logging.debug(
                            f"Profession filtering: {len(self._name_lookup)} → {len(base_df)} records"
                        )
            else:
                # No profession filter
                base_df = self._name_lookup
                print(
                    f"[WARNING] No profession filter - restricting to token-based candidates "
                    f"from ALL {len(base_df):,} records!"
                )

            # 4) Restringi base_df ai soli candidati per token
            candidate_index = pd.Index(candidate_indices)
            search_df = base_df.loc[base_df.index.intersection(candidate_index)]

            if search_df.empty:
                logging.info(
                    f"Fuzzy search: no candidates left after profession+token filtering "
                    f"for '{normalized_name}'"
                )
                return []

            # 5) Crea search_strings (come prima)
            search_strings = [normalized_name]
            if search_professions:
                for prof in search_professions:
                    search_strings.append(f"{normalized_name} {prof}")
                search_strings = list(set(search_strings))

            # 6) Fuzzy matching
            matches_with_scores: List[Dict[str, Any]] = []

            logging.info(f"Starting fuzzy matching on {len(search_df)} records...")
            processed = 0

            for _, row in search_df.iterrows():
                processed += 1
                if processed % 10000 == 0:
                    logging.info(
                        f"  Fuzzy matching progress: {processed}/{len(search_df)} records..."
                    )

                imdb_search_combos = row.get('search_combinations', [])
                if not isinstance(imdb_search_combos, list):
                    imdb_search_combos = [row.get('normalizedName', '')]

                best_similarity = 0
                best_match_string = None

                for query_str in search_strings:
                    for imdb_str in imdb_search_combos:
                        if not imdb_str:
                            continue

                        # micro filtro banale per evitare confronti inutili
                        if abs(len(imdb_str) - len(query_str)) > 10:
                            continue

                        similarity = fuzz.token_sort_ratio(query_str, imdb_str)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match_string = imdb_str

                if best_similarity >= threshold:
                    match_dict = row.to_dict()
                    match_dict['fuzzy_similarity'] = best_similarity
                    match_dict['matched_string'] = best_match_string
                    matches_with_scores.append(match_dict)

            # 7) Ordina per similitudine
            matches_with_scores.sort(
                key=lambda x: x.get('fuzzy_similarity', 0), reverse=True
            )
            return matches_with_scores[:max_results]

        except Exception as e:
            logging.error(f"Fuzzy search error: {e}", exc_info=True)
            return []

    def _has_profession_overlap(self, match_professions: List[str],
                                expected_professions: List[str]) -> bool:
        """Check if professions overlap"""
        if not match_professions or not expected_professions:
            return False

        match_set = {p.lower().strip() for p in match_professions}
        expected_set = {p.lower().strip() for p in expected_professions}

        return bool(match_set & expected_set)

    def _extract_professions(self, match: Dict[str, Any]) -> List[str]:
        """Extract professions from IMDB match"""
        professions_str = match.get('primaryProfession', '')
        if pd.isna(professions_str) or not professions_str:
            return []
        return [p.strip() for p in str(professions_str).split(',')]

    def _apply_assignment_logic(self, original_name: str, normalized_name: str,
                                role_group: Optional[str],
                                matches: List[Dict[str, Any]],
                                is_fuzzy: bool) -> Dict[str, Any]:
        """
        Apply assignment logic - returns ONLY IMDB nconst if unambiguous, NULL otherwise
        """
        logging.info(
            f"Applying assignment logic for '{original_name}' with {len(matches)} "
            f"matches (fuzzy: {is_fuzzy})"
        )

        # Check profession mapping
        if not role_group or not config.has_imdb_profession_mapping(role_group):
            # No profession mapping - assign first match if only one exists
            # This is the behavior for "Thanks", "Additional Crew", etc.
            logging.info(
                f"No profession mapping for '{role_group}' - assigning first "
                f"match if unambiguous"
            )
            if len(matches) == 1:
                match = matches[0]
                return {
                    'assigned_code': match['nconst'],
                    'status': 'auto_assigned',
                    'corrected_name': match['primaryName'],
                    'method': 'no_profession_mapping_single_match'
                }
            else:
                # Multiple matches without profession filter - cannot decide
                logging.info(
                    f"Multiple matches ({len(matches)}) without profession mapping "
                    f"- returning NULL"
                )
                return {
                    'assigned_code': None,
                    'status': 'no_profession_mapping',
                    'corrected_name': None,
                    'method': 'no_profession_mapping_multiple_matches'
                }

        # Get expected professions
        expected_profs = list(config.get_imdb_professions_for_role_group(role_group))
        logging.info(f"Expected professions for '{role_group}': {expected_profs}")

        # Filter by profession compatibility
        compatible: List[Dict[str, Any]] = []
        for match in matches:
            match_profs = self._extract_professions(match)
            if self._has_profession_overlap(match_profs, expected_profs):
                compatible.append(match)

        logging.info(
            f"Found {len(compatible)} compatible matches out of {len(matches)}"
        )

        total = len(matches)
        compatible_count = len(compatible)

        # RULE 1: Single exact match with compatible profession → ASSIGN
        if total == 1 and compatible_count == 1:
            match = compatible[0]
            method = 'single_fuzzy_match' if is_fuzzy else 'single_exact_match'
            logging.info(
                f"Single match with compatible profession: {match['nconst']} "
                f"({match['primaryName']})"
            )
            return {
                'assigned_code': match['nconst'],
                'status': 'auto_assigned',
                'corrected_name': match['primaryName'],
                'method': method
            }

        # RULE 2: Multiple matches, only one compatible → ASSIGN
        elif total > 1 and compatible_count == 1:
            match = compatible[0]
            method = 'multiple_fuzzy_one_compatible' if is_fuzzy \
                else 'multiple_matches_one_compatible'
            logging.info(
                f"Multiple matches but only one compatible: {match['nconst']} "
                f"({match['primaryName']})"
            )
            return {
                'assigned_code': match['nconst'],
                'status': 'auto_assigned',
                'corrected_name': match['primaryName'],
                'method': method
            }

        # RULE 3: Multiple compatible → AMBIGUOUS → NULL
        elif compatible_count > 1:
            logging.info(
                f"Ambiguous: {compatible_count} matches with compatible professions "
                f"→ NULL"
            )
            return {
                'assigned_code': None,
                'status': 'ambiguous',
                'corrected_name': None,
                'method': 'multiple_compatible_matches'
            }

        # RULE 6: In IMDB but no compatible profession → NULL
        elif total > 0 and compatible_count == 0:
            logging.info(
                f"Found in IMDB but no compatible profession → NULL"
            )
            return {
                'assigned_code': None,
                'status': 'incompatible_profession',
                'corrected_name': None,
                'method': 'incompatible_profession'
            }

        # Fallback → NULL
        else:
            logging.warning(
                f"Unexpected case: total={total}, compatible={compatible_count} "
                f"→ NULL"
            )
            return {
                'assigned_code': None,
                'status': 'unexpected',
                'corrected_name': None,
                'method': 'unexpected_case'
            }

    def validate_name(self, name: str, role_group: Optional[str] = None, threshold: Optional[int] = None,
                       normalized_name_with_nickname: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate name against IMDB - returns ONLY IMDB matches, NULL if not found.
        NO internal codes generation - just IMDB nconst or None.

        normalized_name_with_nickname: pre-computed with-nickname normalized
        form (see normalize_name_with_nickname), or None if `name` has no
        quoted nickname aside. When given, an exact match against it is
        tried FIRST; only falls through to the regular (nickname-stripped)
        flow below if that has no hits at all.
        """
        if self._name_lookup is None:
            return {
                'assigned_code': None,
                'status': 'no_imdb_data',
                'corrected_name': None,
                'method': 'no_imdb_data'
            }

        # Use provided threshold or default
        current_threshold = threshold if threshold is not None else self.fuzzy_threshold

        original_name = name

        if normalized_name_with_nickname:
            nickname_matches = self._exact_match_imdb_with_nickname(normalized_name_with_nickname)
            if nickname_matches:
                nickname_result = self._apply_assignment_logic(
                    original_name, normalized_name_with_nickname, role_group, nickname_matches, is_fuzzy=False
                )
                if nickname_result['status'] == 'auto_assigned':
                    logging.info(
                        f"Found {len(nickname_matches)} exact match(es) via with-nickname key "
                        f"'{normalized_name_with_nickname}' for '{original_name}' -> "
                        f"auto-assigned {nickname_result['assigned_code']}"
                    )
                    return nickname_result
                logging.info(
                    f"With-nickname key '{normalized_name_with_nickname}' matched but didn't resolve "
                    f"decisively ({nickname_result['status']}) for '{original_name}' - "
                    f"falling back to the nickname-stripped flow"
                )

        # Extract Jr./Sr. suffix BEFORE normalization (look for ", Jr." or ", Sr." pattern)
        suffix = None
        suffix_match = re.search(r',\s*(Jr\.?|Sr\.?)\s*$', original_name, re.IGNORECASE)
        if suffix_match:
            suffix = suffix_match.group(1).lower().rstrip('.')
            # Remove the suffix from name before normalization
            name_for_normalization = original_name[:suffix_match.start()].strip()
            logging.debug(f"Extracted suffix '{suffix}' from original name '{original_name}', normalizing '{name_for_normalization}'")
        else:
            name_for_normalization = original_name
        
        # is_person=True: this validator only ever runs on person rows (the
        # caller skips companies before calling validate_name)
        normalized = normalize_name(name_for_normalization, is_person=True)

        # Split into words for permutation testing (EXACT same logic)
        words = normalized.split()

        if len(words) == 0:
            logging.warning(f"No valid words in '{normalized}'")
            return {
                'assigned_code': None,
                'status': 'no_valid_words',
                'corrected_name': None,
                'method': 'no_valid_words'
            }

        # Safeguard: detect concatenated names (EXACT same logic)
        if len(words) > 6:
            logging.warning(
                f"'{original_name}' has {len(words)} words - likely concatenated"
            )
            return {
                'assigned_code': None,
                'status': 'concatenated_names',
                'corrected_name': None,
                'method': 'concatenated_names_detected'
            }

        # Generate permutations (EXACT same logic with factorial limit)
        max_permutations = 720  # 6! = 720
        if math.factorial(len(words)) > max_permutations:
            logging.warning(
                f"'{original_name}' would generate {math.factorial(len(words))} "
                f"permutations - using only original order"
            )
            base_permutations = [' '.join(words)]
        else:
            base_permutations = set([' '.join(p) for p in itertools.permutations(words)])
        
        # If we extracted a suffix (Jr./Sr.), append it to all permutations
        if suffix:
            permutations = set([f"{perm} {suffix}" for perm in base_permutations])
        else:
            permutations = base_permutations

        logging.info(
            f"IMDB validation for '{original_name}' -> normalized: '{normalized}' "
            f"-> words: {words}{' + suffix: ' + suffix if suffix else ''} -> {len(permutations)} permutations"
        )

        # Try exact matches for ALL permutations (EXACT same logic but usando l'indice)
        exact_matches: List[Dict[str, Any]] = []
        for perm in permutations:
            logging.debug(f"Searching for exact match: '{perm}'")
            matches = self._exact_match_imdb(perm)
            if matches:
                logging.info(
                    f"Found {len(matches)} exact match(es) for permutation '{perm}'"
                )
                exact_matches.extend(matches)

        # Determine if we found matches
        if exact_matches:
            matches = exact_matches
            is_fuzzy = False
            logging.info(
                f"Found {len(exact_matches)} exact match(es) for '{original_name}'"
            )
        elif self.fuzzy_enabled and current_threshold < 100:
            # Check if we should skip fuzzy for certain role groups
            if role_group and role_group.lower() in ['thanks', 'unknown']:
                # No fuzzy matching for Thanks/Unknown - only exact matches (no profession mapping)
                logging.info(
                    f"Skipping fuzzy matching for role_group '{role_group}' "
                    f"- exact match only"
                )
                matches = []
                is_fuzzy = False
            else:
                # Try fuzzy matching (EXACT same logic, ma con indice)
                logging.info(
                    f"No exact matches for '{original_name}', trying fuzzy "
                    f"(threshold: {current_threshold}%)"
                )
                matches = self._fuzzy_search_imdb(
                    normalized, role_group, current_threshold
                )
                is_fuzzy = True
                if matches:
                    logging.info(
                        f"Found {len(matches)} fuzzy match(es) with similarity "
                        f">= {current_threshold}%"
                    )
        else:
            matches = []
            is_fuzzy = False

        # No matches - return NULL
        if not matches:
            logging.info(f"No IMDB matches for '{original_name}'")
            return {
                'assigned_code': None,
                'status': 'no_match',
                'corrected_name': None,
                'method': 'no_match'
            }

        # Apply assignment logic (EXACT same as original)
        return self._apply_assignment_logic(
            original_name, normalized, role_group, matches, is_fuzzy
        )


def imdbize_human_data():
    """
    Apply IMDB validation to human-corrected credits using identical logic
    as the automated pipeline, with performance optimizations.
    """
    print("=" * 60)
    print("STARTING IMDBIZATION PROCESS")
    print("=" * 60)

    # Input/output paths
    input_dir = Path(__file__).parent / "human_data_to_be_imdbized"
    input_file = input_dir / "credits_human_corrected_to_be_imdbized_validation_5.csv"
    output_file = input_dir / "IMDBIZED_credits_human_corrected_validation_5.csv"

    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    if not input_file.exists():
        logging.error(f"❌ Input file not found: {input_file}")
        return

    # input_file is ALWAYS the source of the rows to (re)normalize and
    # (re)match - it never has the HUMAN correction columns at all. Those
    # only ever live in output_file, added by hand after a previous run.
    # If output_file already exists: back it up, then load it (no more
    # interactive yes/no prompt - this now always happens automatically)
    # purely to pull its HUMAN columns back out and reinsert them into the
    # freshly-recomputed data below, matched by (nome, numero_episodio).
    existing_df = None
    if output_file.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = output_file.with_name(f"{output_file.stem}_backup_{timestamp}{output_file.suffix}")
        backup_path.write_bytes(output_file.read_bytes())
        print(f"💾 Backed up existing output to: {backup_path}")

        print(f"📖 Loading existing output for HUMAN column reinsertion...")
        existing_df = pd.read_csv(output_file, sep=';', encoding='utf-8-sig')
        existing_df.columns = existing_df.columns.str.strip()
        if '' in existing_df.columns:
            existing_df = existing_df.drop(columns=[''])
        print(f"✅ Loaded existing output with {len(existing_df)} rows")

    print(f"📖 Loading CSV file...")
    logging.info(f"📖 Loading human-corrected credits from: {input_file}")
    df = pd.read_csv(input_file, sep=';', encoding='utf-8')  # Semicolon separator
    print(f"✅ Loaded {len(df)} rows")

    # Strip whitespace from column names and drop empty columns
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Remove trailing semicolon column if present (empty column)
    if '' in df.columns:
        df = df.drop(columns=[''])

    print(f"Columns: {list(df.columns)}")
    logging.info(f"✅ Loaded {len(df)} credits")
    logging.info(f"Columns: {list(df.columns)}")

    # Verify required columns
    required_cols = ['nome', 'is_person', 'role_group']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"❌ Missing required columns: {missing_cols}")
        return

    # OPTIMIZATION: Sort by role_group to group same professions together
    print(f"📊 Sorting by role_group for optimal profession filtering...")
    df = df.sort_values(by='role_group', na_position='last')
    df = df.reset_index(drop=True)  # CRITICAL: Reset indices after sorting!
    print(f"✅ Sorted - same professions will be processed together")

    # Stessa pipeline is_person-aware usata ovunque nel progetto (save_credits,
    # compare_llm_human_metrics.py, compare_gpt_sol_standard_validation5.py):
    # per le persone toglie anche virgolette/parentesi, per le aziende le
    # conserva (es. "RAI" vs "RAI (Roma)" devono restare distinte).
    print(f"🧹 Computing normalized_name column...")
    df['normalized_name'] = df.apply(
        lambda r: normalize_name(str(r['nome']), is_person=(str(r.get('is_person')).upper() == 'TRUE'))
        if pd.notna(r['nome']) else None,
        axis=1
    )
    print(f"✅ normalized_name column added")

    # Companion column: same source, but KEEPS a quoted nickname aside
    # (e.g. 'roy "bucky" moore' instead of 'roy moore') - None when there is
    # no such aside, or for companies. A nickname can be the only thing that
    # disambiguates a common name among many IMDB namesakes once it's
    # stripped, so validate_name() below tries this first.
    print(f"🧹 Computing normalized_name_with_nickname column...")
    df['normalized_name_with_nickname'] = df.apply(
        lambda r: normalize_name_with_nickname(str(r['nome']), is_person=(str(r.get('is_person')).upper() == 'TRUE'))
        if pd.notna(r['nome']) else None,
        axis=1
    )
    print(f"✅ normalized_name_with_nickname column added")

    # Initialize validator
    print("🔧 Initializing IMDB validator...")
    logging.info("🔧 Initializing standalone IMDB validator (fuzzy=True, threshold=90%)...")
    validator = StandaloneIMDBValidator(fuzzy_enabled=True, fuzzy_threshold=90)
    print("✅ Validator initialized")

    if validator._name_lookup is None:
        print("❌ IMDB database not loaded!")
        logging.error("❌ IMDB database not loaded! Cannot proceed.")
        return

    print(f"✅ IMDB database loaded: {len(validator._name_lookup)} records")

    # Add new columns for IMDB results
    THRESHOLDS = [88, 90, 92, 94, 96, 98, 99]
    
    for t in THRESHOLDS:
        df[f'imdb_nconst_{t}'] = None
        df[f'nome_corretto_imdb_{t}'] = None
        df[f'imdb_action_{t}'] = None

    # We never skip already-processed rows or carry forward their old automatic
    # imdb_nconst_*/nome_corretto_imdb_*/imdb_action_* values: the IMDB lookup
    # is always redone for every row against the freshly recomputed
    # normalized_name/normalized_name_with_nickname above, since that can
    # surface matches the previous (less thorough) normalization missed. Only
    # the manually-entered HUMAN columns are preserved, pulled back out of the
    # (already backed-up) existing output file loaded above.
    if existing_df is not None:
        human_cols = ['HUMAN nome imdb corretto', 'HUMAN codice imdb corretto']
        existing_human_cols = [c for c in human_cols if c in existing_df.columns]
        if existing_human_cols and 'numero_episodio' in existing_df.columns and 'numero_episodio' in df.columns:
            key_cols = ['nome', 'numero_episodio']
            human_lookup = existing_df.loc[
                existing_df[existing_human_cols].notna().any(axis=1), key_cols + existing_human_cols
            ]
            dupe_keys = human_lookup.duplicated(subset=key_cols, keep=False)
            if dupe_keys.any():
                print(f"⚠️  {dupe_keys.sum()} existing HUMAN-corrected rows share a duplicate "
                      f"(nome, numero_episodio) key - skipping those to avoid an ambiguous merge")
                human_lookup = human_lookup[~dupe_keys]
            df = df.merge(human_lookup, on=key_cols, how='left')
            print(f"✅ Preserved {len(human_lookup)} existing HUMAN correction(s) by (nome, numero_episodio)")
            logging.info(f"Preserved {len(human_lookup)} HUMAN corrections from existing output")

    for c in ['HUMAN nome imdb corretto', 'HUMAN codice imdb corretto']:
        if c not in df.columns:
            df[c] = None

    print(f"🔍 Starting to process {len(df)} credits...")
    logging.info("🔍 Processing credits with IMDB validation...")

    # Stats
    stats = {
        'total': 0,
        'persons_processed': 0
    }

    # Salvataggio ogni N righe (rimane 50 come richiesto)
    SAVE_INTERVAL = 50
    rows_since_last_save = 0

    # Cache di alto livello per evitare di rifare validate_name su stessi (nome_normalizzato, role_group, threshold)
    validation_cache: Dict[Any, Dict[str, Any]] = {}

    for idx, row in df.iterrows():
        name = row['nome']
        is_person = row['is_person']
        role_group = row.get('role_group', None)

        stats['total'] += 1

        # Skip se non persona
        if not is_person or pd.isna(is_person) or str(is_person).upper() != 'TRUE':
            logging.debug(f"Row {idx}: Skipping non-person entity: {name}")
            continue

        # NOTA: nessuno skip per riga "già processata" - l'IMDB lookup viene
        # sempre rifatto su normalized_name per ogni riga, anche in resume mode
        # (solo le colonne HUMAN vengono preservate, non i risultati automatici).

        stats['persons_processed'] += 1

        print(f"\n--- Processing row {idx+1}/{len(df)} ---")
        print(f"Name: {name}, Role: {role_group}, IsPerson: {is_person}")

        # Skip IMDB search for "Thanks" and "Unknown" - always NULL (no code assigned in human data)
        if role_group and isinstance(role_group, str) and role_group.lower() in ['thanks', 'unknown']:
            print(f"[SKIP] '{name}' - {role_group} role (no IMDB search)")
            logging.info(f"Row {idx}: Skipping IMDB search for {role_group} role: '{name}'")
            for t in THRESHOLDS:
                df.at[idx, f'imdb_action_{t}'] = 'X'
                df.at[idx, f'nome_corretto_imdb_{t}'] = name
            continue

        # Nome ripulito da onorifici/virgolette/parentesi ma non ancora passato
        # per l'ultimo normalize_name: lo passiamo a validate_name cosi' la sua
        # estrazione del suffisso Jr./Sr. (che cerca una virgola) funziona
        # ancora; il normalize_name() interno di validate_name produce lo
        # stesso valore gia' calcolato in df['normalized_name'].
        pre_normalized = strip_parentheticals(strip_quoted_asides(strip_honorifics(str(name))))
        norm_name = row['normalized_name']  # chiave di cache = nome normalizzato finale
        nickname_name = row.get('normalized_name_with_nickname')
        if pd.isna(nickname_name):
            nickname_name = None
        role_key = role_group.lower() if isinstance(role_group, str) else None

        # Loop over thresholds
        for t in THRESHOLDS:
            cache_key = (norm_name, nickname_name, role_key, t)

            # Validate (con cache)
            try:
                if cache_key in validation_cache:
                    result = validation_cache[cache_key]
                else:
                    result = validator.validate_name(
                        name=pre_normalized, role_group=role_group, threshold=t,
                        normalized_name_with_nickname=nickname_name
                    )
                    validation_cache[cache_key] = result

                # Store IMDB results
                imdb_code = result['assigned_code']
                imdb_name = result['corrected_name']

                df.at[idx, f'imdb_nconst_{t}'] = imdb_code

                if imdb_code and imdb_name:
                    # Found in IMDB - always set the IMDB name
                    df.at[idx, f'nome_corretto_imdb_{t}'] = imdb_name

                    # Check if name was modified (person-only loop, so is_person=True)
                    if normalize_name(imdb_name, is_person=True) != normalize_name(name, is_person=True):
                        df.at[idx, f'imdb_action_{t}'] = 'M'
                        if t == 90: # Log only for standard threshold to avoid spam
                            print(f"[M][{t}] '{name}' → '{imdb_name}' ({imdb_code})")
                    else:
                        df.at[idx, f'imdb_action_{t}'] = 'A'
                        if t == 90:
                            print(f"[A][{t}] '{name}' = '{imdb_name}' ({imdb_code})")
                else:
                    # Not found in IMDB
                    df.at[idx, f'imdb_action_{t}'] = 'X'
                    df.at[idx, f'nome_corretto_imdb_{t}'] = name  # Keep original
                    status = result['status']
                    if t == 90:
                        print(f"[X][{t}] '{name}' - {status}")

            except Exception as e:
                logging.error(f"❌ Error processing row {idx} ('{name}') threshold {t}: {e}", exc_info=True)
                continue

        # Incremental save (rimasto a 50 righe)
        rows_since_last_save += 1
        if rows_since_last_save >= SAVE_INTERVAL:
            print(f"\n💾 Saving progress... ({idx + 1}/{len(df)} rows processed)")
            df.to_csv(output_file, sep=';', index=False, encoding='utf-8-sig')
            rows_since_last_save = 0

        # Progress indicator
        if (idx + 1) % 100 == 0:
            logging.info(f"Progress: {idx + 1}/{len(df)} rows processed...")

    # Save output finale
    logging.info(f"💾 Saving IMDBized credits to: {output_file}")
    df.to_csv(output_file, sep=';', index=False, encoding='utf-8-sig')

    # Print statistics
    logging.info("\n" + "=" * 60)
    logging.info("📊 IMDBIZATION STATISTICS")
    logging.info("=" * 60)
    logging.info(f"Total rows in file: {stats['total']}")
    logging.info(f"Persons processed: {stats['persons_processed']}")
    logging.info("=" * 60)
    logging.info(f"✅ Output saved to: {output_file}")
    logging.info("=" * 60)


if __name__ == "__main__":
    imdbize_human_data()
