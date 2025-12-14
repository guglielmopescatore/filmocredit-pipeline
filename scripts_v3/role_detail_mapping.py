"""
Role Detail to Role Group Hard Mapping

This module provides automatic correction of LLM-assigned role_group based on role_detail keywords.
When enabled, it overrides incorrect LLM categorizations using EXACT pattern matching on role_detail text.

The mapping is loaded from an external JSON file for easier maintenance.
JSON structure: { "Role Group": ["role_detail_1", "role_detail_2", ...], ... }
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Set


# Path to the JSON mapping file
_MAPPING_FILE_PATH = Path(__file__).resolve().parent.parent / 'db' / 'role_groups_mapping_unique_lowercase.json'


# Cache for the loaded mapping - direct structure from JSON
# Structure: { "Role Group": set(["role_detail_1", "role_detail_2", ...]), ... }
_role_group_to_details_cache: Optional[Dict[str, Set[str]]] = None


def _load_mapping_from_json() -> Dict[str, Set[str]]:
    """
    Load the role group mapping from the external JSON file.
    
    The JSON file has structure:
    {
        "Role Group Name": ["role_detail_1", "role_detail_2", ...],
        ...
    }
    
    Returns:
        Dict mapping role_group to set of lowercase role_detail strings
    """
    global _role_group_to_details_cache
    
    if _role_group_to_details_cache is not None:
        return _role_group_to_details_cache
    
    try:
        if not _MAPPING_FILE_PATH.exists():
            logging.warning(f"Role mapping file not found: {_MAPPING_FILE_PATH}")
            _role_group_to_details_cache = {}
            return _role_group_to_details_cache
        
        with open(_MAPPING_FILE_PATH, 'r', encoding='utf-8') as f:
            raw_mapping: Dict[str, List[str]] = json.load(f)
        
        # Convert lists to sets of normalized (lowercase) role_details
        normalized_mapping: Dict[str, Set[str]] = {}
        total_details = 0
        for role_group, role_details in raw_mapping.items():
            normalized_set = set()
            for role_detail in role_details:
                normalized = role_detail.lower().strip()
                if normalized:
                    normalized_set.add(normalized)
            normalized_mapping[role_group] = normalized_set
            total_details += len(normalized_set)
        
        _role_group_to_details_cache = normalized_mapping
        logging.info(f"Loaded mapping with {len(normalized_mapping)} role groups and {total_details} role details from {_MAPPING_FILE_PATH}")
        return _role_group_to_details_cache
        
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing role mapping JSON file: {e}")
        _role_group_to_details_cache = {}
        return _role_group_to_details_cache
    except Exception as e:
        logging.error(f"Error loading role mapping file: {e}")
        _role_group_to_details_cache = {}
        return _role_group_to_details_cache


def get_role_detail_mapping() -> Dict[str, Set[str]]:
    """
    Get the role group to role details mapping.
    
    Returns:
        Dict mapping role_group to set of lowercase role_detail strings
    """
    return _load_mapping_from_json()


def find_role_group_for_detail(normalized_detail: str) -> Optional[str]:
    """
    Find which role_group contains a given role_detail.
    
    Args:
        normalized_detail: Lowercase, stripped role_detail string
        
    Returns:
        The role_group containing this detail, or None if not found
    """
    mapping = get_role_detail_mapping()
    for role_group, details_set in mapping.items():
        if normalized_detail in details_set:
            return role_group
    return None


def reload_mapping() -> Dict[str, Set[str]]:
    """
    Force reload the mapping from the JSON file.
    Useful when the JSON file has been updated.
    
    Returns:
        Dict mapping role_group to set of role_detail strings
    """
    global _role_group_to_details_cache
    _role_group_to_details_cache = None
    return _load_mapping_from_json()


def normalize_role_detail(role_detail: Optional[str]) -> str:
    """
    Normalize role_detail for matching using the SAME logic as mapping creation.
    
    Normalization steps:
    1. Fix mojibake (latin1 -> utf-8)
    2. Convert to string
    3. Strip whitespace
    4. Convert to lowercase
    
    Returns:
        Normalized role_detail string
    """
    if not role_detail:
        return ""
    
    # Step 1: Fix mojibake (same as mapping creation)
    try:
        if isinstance(role_detail, str):
            role_detail = role_detail.encode("latin1").decode("utf-8")
    except Exception:
        # If mojibake fix fails, continue with original
        pass
    
    # Steps 2-4: Convert to string, strip, lowercase
    return str(role_detail).strip().lower()


def correct_role_group_from_detail(
    role_detail: Optional[str],
    current_role_group: str,
    enabled: bool = False,
    is_person: bool = True
) -> Optional[str]:
    """
    Apply hard mapping correction based on EXACT role_detail match.
    
    IMPORTANT: Mapping is applied ONLY if:
    - enabled is True
    - is_person is True (no mapping for companies)
    - current_role_group is NOT "Cast" (Cast roles are never remapped)
    
    Args:
        role_detail: The role detail text from LLM
        current_role_group: The role_group assigned by LLM
        enabled: Whether correction is enabled (default False - disabled)
        is_person: Whether this credit is for a person (default True)
        
    Returns:
        Corrected role_group if EXACT pattern matches, None otherwise
    """
    if not enabled or not role_detail or not is_person:
        return None
    
    # NEVER remap Cast - Cast assignments are always preserved
    if current_role_group == "Cast":
        return None
    
    normalized_detail = normalize_role_detail(role_detail)
    if not normalized_detail:
        return None
    
    # Find role_group that contains this role_detail
    target_role_group = find_role_group_for_detail(normalized_detail)
    
    if target_role_group and target_role_group != current_role_group:
        logging.debug(
            f"Role correction: '{role_detail}' -> '{target_role_group}' "
            f"(was: '{current_role_group}')"
        )
        return target_role_group
    
    # No match found or same as current - no correction needed
    return None


def apply_role_corrections_to_credits(
    credits: List[Dict],
    enabled: bool = False
) -> tuple[List[Dict], int]:
    """
    Apply role_group corrections to a list of credits from LLM output.
    Adds 'role_group_corrected' field when correction is applied.
    
    Mapping is applied ONLY for:
    - is_person == True (no mapping for companies)
    - role_group != "Cast" (Cast roles are never remapped)
    
    Args:
        credits: List of credit dictionaries from LLM
        enabled: Whether correction is enabled (default False - disabled)
        
    Returns:
        Tuple of (corrected_credits, correction_count)
    """
    if not enabled:
        logging.debug("Role correction disabled, skipping")
        return credits, 0
    
    # Ensure mapping is loaded
    mapping = get_role_detail_mapping()
    logging.info(f"Role correction enabled. Mapping has {len(mapping)} entries. Processing {len(credits)} credits.")
    
    correction_count = 0
    corrected_credits = []
    
    for credit in credits:
        credit_copy = credit.copy()
        role_detail = credit.get('role_detail')
        current_role_group = credit.get('role_group', '')
        is_person = credit.get('is_person', True)  # Default to True if not specified
        
        # Debug: log what we're checking
        normalized = normalize_role_detail(role_detail) if role_detail else ""
        in_mapping = normalized in mapping if normalized else False
        logging.debug(
            f"Checking credit: role_detail='{role_detail}' normalized='{normalized}' "
            f"current_group='{current_role_group}' is_person={is_person} in_mapping={in_mapping}"
        )
        
        corrected_role_group = correct_role_group_from_detail(
            role_detail, current_role_group, enabled=True, is_person=is_person
        )
        
        if corrected_role_group:
            credit_copy['role_group'] = corrected_role_group
            credit_copy['role_group_corrected'] = True
            credit_copy['original_role_group'] = current_role_group
            correction_count += 1
            logging.info(
                f"CORRECTED: '{role_detail}' from '{current_role_group}' to '{corrected_role_group}'"
            )
        else:
            credit_copy['role_group_corrected'] = False
        
        corrected_credits.append(credit_copy)
    
    logging.info(f"Role correction complete: {correction_count} corrections out of {len(credits)} credits")
    
    return corrected_credits, correction_count


# Backward compatibility alias
def get_corrected_role_group(role_detail: str, enabled: bool = False) -> Optional[str]:
    """
    Get the corrected role group for a given role detail.
    
    This is a backward-compatible wrapper around correct_role_group_from_detail.
    
    Args:
        role_detail: The role detail text to look up
        enabled: Whether correction is enabled (default False)
        
    Returns:
        The corrected role_group if found and enabled, None otherwise
    """
    if not enabled:
        return None
    return correct_role_group_from_detail(role_detail, "", enabled=True)
