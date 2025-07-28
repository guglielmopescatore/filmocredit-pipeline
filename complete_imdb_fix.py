#!/usr/bin/env python3
"""
Complete IMDB fix script that processes ALL credits:
- Credits found in IMDB: Update with profession data and apply corrected logic
- Credits NOT found in IMDB: Assign internal gp codes
"""

import sqlite3
import json
import pandas as pd
from pathlib import Path
import sys
import os
import re
import unicodedata

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def normalize_name(name):
    """Normalize name for IMDB matching"""
    if not isinstance(name, str):
        name = str(name)
    
    # Convert to lowercase
    name = name.lower()
    
    # Remove titles
    name = re.sub(
        r"\b("
        r"dr\.|dott\.|dott\.ssa|prof\.|prof\.ssa|ing\.|arch\.|avv\.|sig\.|sig\.ra|sig\.na|"
        r"mr\.|mrs\.|ms\.|mx\.|fr\.|rev\.|hon\.|sen\.|rep\.|gov\.|pres\.|vp\.|"
        r"capt\.|cmdr\.|lt\.|col\.|maj\.|gen\.|adm\.|"
        r"msgr\.|sr\.|sra\.|srta\.|srs\.|"
        r"mlle\.|mme\.|mons\.|pr\.|amb\.|pm\.|"
        r"ph\.?d|m\.?d|esq\.|emo\.|eccmo\.|p\.i|geom\."
        r")\s+",
        "", 
        name,
        flags=re.IGNORECASE
    )
    
    # Normalize unicode to decompose accented characters
    name = unicodedata.normalize('NFD', name)
    # Remove diacritical marks (accents)
    name = ''.join(char for char in name if unicodedata.category(char) != 'Mn')
    
    # Remove punctuation but preserve spaces
    name = re.sub(r"[.\-_,']", "", name)
    
    # Remove any extra whitespace and normalize to single spaces
    name = re.sub(r"\s+", " ", name).strip()
    
    return name

def get_imdb_professions_for_role_group(role_group):
    """Get expected IMDB professions for a role group"""
    profession_mappings = {
        'Cast': ['actor', 'actress'],
        'Directors': ['director'],
        'Writers': ['writer'],
        'Producers': ['producer'],
        'Cinematographers': ['cinematographer', 'director_of_photography'],
        'Editors': ['editor'],
        'Composers': ['composer', 'music_department'],
        'Production Designers': ['production_designer', 'art_department'],
        'Art Directors': ['art_director', 'art_department'],
        'Set Decorators': ['set_decorator', 'art_department'],
        'Costume Designers': ['costume_designer'],
        'Makeup Department': ['make_up', 'makeup_department'],
        'Sound Department': ['sound_department', 'sound_mixer'],
        'Visual Effects': ['visual_effects', 'special_effects'],
        'Special Effects': ['special_effects'],
        'Music Department': ['music_department'],
        'Production Managers': ['production_manager'],
        'Location Managers': ['location_manager'],
        'Casting Directors': ['casting_director'],
        'Second Unit Directors or Assistant Directors': ['assistant_director'],
        'Camera and Electrical Department': ['camera_department'],
        'Art Department': ['art_department'],
        'Animation Department': ['animation_department'],
        'Costume and Wardrobe Department': ['costume_department'],
        'Editorial Department': ['editorial_department'],
        'Script and Continuity Department': ['script_department'],
        'Transportation Department': ['transportation_department'],
        'Stunts': ['stunt'],
        'Thanks': ['thanks'],
        'Additional Crew': ['additional_crew']
    }
    return profession_mappings.get(role_group, [])

def is_profession_compatible(imdb_professions, expected_professions):
    """Check if IMDB professions are compatible with expected role group professions"""
    if not imdb_professions or not expected_professions:
        return False
    
    imdb_profession_list = imdb_professions.lower().split(',')
    expected_profession_list = [p.lower() for p in expected_professions]
    
    # Check if any IMDB profession matches any expected profession
    for imdb_prof in imdb_profession_list:
        imdb_prof = imdb_prof.strip()
        for expected_prof in expected_profession_list:
            if imdb_prof == expected_prof or imdb_prof in expected_prof or expected_prof in imdb_prof:
                return True
    return False

def check_existing_internal_code(normalized_name, role_group, is_company, main_conn=None):
    """Check if we already have an internal code for this normalized name and role."""
    try:
        # Always use a separate connection to avoid locking issues
        conn = sqlite3.connect('db/tvcredits_v3.db')
        cursor = conn.cursor()
        
        # Determine the code prefix to search for
        code_prefix = 'cm' if is_company else 'gp'
        
        # Search for existing credits with the same normalized name, role group, and internal code type
        cursor.execute("""
            SELECT assigned_code, name, role_group_normalized
            FROM credits 
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
                    print(f"🔄 Reusing existing internal code '{assigned_code}' for '{normalized_name}' in role '{role_group}'")
                    return assigned_code
        
        return None
        
    except Exception as e:
        print(f"Error checking existing internal codes: {e}")
        return None


def generate_next_internal_code(is_company=False):
    """Generate next internal code (gp for persons, cm for companies)"""
    conn = sqlite3.connect('db/tvcredits_v3.db')
    cursor = conn.cursor()
    
    prefix = 'cm' if is_company else 'gp'
    
    # Find the highest existing code
    cursor.execute(f"""
        SELECT assigned_code 
        FROM credits 
        WHERE assigned_code LIKE '{prefix}%' 
        ORDER BY assigned_code DESC 
        LIMIT 1
    """)
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        last_code = result[0]
        # Extract number and increment
        try:
            number = int(last_code[2:])  # Remove prefix
            return f"{prefix}{number + 1:07d}"
        except ValueError:
            pass
    
    # Start with 1 if no existing codes
    return f"{prefix}0000001"

def complete_imdb_fix():
    """Complete IMDB fix for ALL credits"""
    
    print("🔄 Starting complete IMDB fix for ALL credits...")
    
    # Load the parquet file
    parquet_path = Path('db/normalized_names.parquet')
    if not parquet_path.exists():
        print("❌ Parquet file not found!")
        return
    
    print("📊 Loading parquet file...")
    df = pd.read_parquet(parquet_path)
    print(f"✅ Loaded {len(df)} IMDB records with profession data")
    
    # Connect to database with better transaction management
    conn = sqlite3.connect('db/tvcredits_v3.db')
    conn.execute("PRAGMA journal_mode=WAL")  # Use WAL mode for better concurrency
    conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
    cursor = conn.cursor()
    
    # Get ALL credits (except companies which will be skipped)
    cursor.execute("""
        SELECT id, name, role_group_normalized, imdb_matches, assigned_code, code_assignment_status, is_person
        FROM credits 
        ORDER BY name
    """)
    
    credits = cursor.fetchall()
    print(f"📋 Found {len(credits)} credits to process")
    
    updated_count = 0
    auto_assigned_imdb_count = 0
    auto_assigned_internal_count = 0
    manual_required_count = 0
    
    for credit_id, name, role_group, imdb_matches, assigned_code, current_status, is_person in credits:
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Skip companies - they should get internal cm codes automatically
                if is_person is False or is_person == 0:
                    # This is a company - check for existing internal code first
                    normalized_name = normalize_name(name)
                    existing_code = check_existing_internal_code(normalized_name, role_group, is_company=True)
                    
                    if existing_code:
                        # Reuse existing code
                        internal_code = existing_code
                        print(f"🔄 Reusing company code: {name} -> {internal_code} (role: {role_group})")
                    else:
                        # Generate new internal code
                        internal_code = generate_next_internal_code(is_company=True)
                        print(f"✅ Auto-assigned company: {name} -> {internal_code} (role: {role_group})")
                    
                    new_status = 'internal_assigned'
                    auto_assigned_internal_count += 1
                    
                    # Update the database
                    cursor.execute("""
                        UPDATE credits 
                        SET imdb_matches = NULL, assigned_code = ?, code_assignment_status = ?
                        WHERE id = ?
                    """, (internal_code, new_status, credit_id))
                    
                    updated_count += 1
                    break  # Success, exit retry loop
                
                # Normalize the name for IMDB search (only for persons)
                normalized_name = normalize_name(name)
                
                # Search for this person in the IMDB parquet file
                imdb_matches_found = df[df['normalizedName'] == normalized_name]
                
                if not imdb_matches_found.empty:
                    # Person found in IMDB
                    matches = []
                    for _, person in imdb_matches_found.iterrows():
                        match = {
                            'nconst': person['nconst'],
                            'normalized_name': person['normalizedName'],
                            'primaryName': person['primaryName'],
                            'primaryProfession': person['primaryProfession'],
                            'birthYear': person['birthYear'],
                            'deathYear': person['deathYear']
                        }
                        matches.append(match)
                    
                    # Apply the corrected assignment logic
                    new_assigned_code = None
                    new_status = 'manual_required'
                    
                    if matches:
                        # Get expected professions for this role group
                        expected_professions = get_imdb_professions_for_role_group(role_group)
                        
                        # Find compatible matches
                        compatible_matches = []
                        for match in matches:
                            imdb_professions = match.get('primaryProfession', '')
                            if is_profession_compatible(imdb_professions, expected_professions):
                                compatible_matches.append(match)
                        
                        # Apply the corrected logic
                        if len(compatible_matches) == 1:
                            # Exactly ONE compatible match - auto-assign
                            new_assigned_code = compatible_matches[0]['nconst']
                            new_status = 'auto_assigned'
                            auto_assigned_imdb_count += 1
                            print(f"✅ Auto-assigned IMDB: {name} -> {new_assigned_code} (role: {role_group})")
                        elif len(compatible_matches) > 1:
                            # Multiple compatible matches - manual review
                            new_status = 'ambiguous'
                            manual_required_count += 1
                            print(f"⚠️  Multiple matches for {name} (role: {role_group}) - manual review needed")
                        else:
                            # No compatible matches - manual review
                            new_status = 'manual_required'
                            manual_required_count += 1
                            print(f"❌ No compatible matches for {name} (role: {role_group}) - manual review needed")
                    
                    # Update the database with new IMDB matches and status
                    new_imdb_matches = json.dumps(matches)
                    cursor.execute("""
                        UPDATE credits 
                        SET imdb_matches = ?, assigned_code = ?, code_assignment_status = ?
                        WHERE id = ?
                    """, (new_imdb_matches, new_assigned_code, new_status, credit_id))
                    
                else:
                    # Person NOT found in IMDB - assign internal code
                    is_company = (is_person is False or is_person == 0)
                    
                    # Check for existing internal code first
                    existing_code = check_existing_internal_code(normalized_name, role_group, is_company=False)
                    
                    if existing_code:
                        # Reuse existing code
                        internal_code = existing_code
                        print(f"🔄 Reusing person code: {name} -> {internal_code} (role: {role_group})")
                    else:
                        # Generate new internal code
                        internal_code = generate_next_internal_code(is_company)
                        print(f"✅ Auto-assigned internal: {name} -> {internal_code} (role: {role_group})")
                    
                    new_status = 'internal_assigned'
                    auto_assigned_internal_count += 1
                    
                    # Update the database with internal code and clear IMDB matches
                    cursor.execute("""
                        UPDATE credits 
                        SET imdb_matches = NULL, assigned_code = ?, code_assignment_status = ?
                        WHERE id = ?
                    """, (internal_code, new_status, credit_id))
                
                updated_count += 1
                
                if updated_count % 50 == 0:
                    print(f"✅ Processed {updated_count} credits...")
                    # Commit periodically to reduce transaction size
                    conn.commit()
                    
                break  # Success, exit retry loop
                
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"⚠️  Retry {retry_count}/{max_retries} for credit {credit_id}: {e}")
                    import time
                    time.sleep(0.1)  # Small delay before retry
                else:
                    print(f"❌ Error processing credit {credit_id} after {max_retries} retries: {e}")
                    break
    
    # Commit changes
    conn.commit()
    conn.close()
    
    print(f"\n🎉 Successfully processed {updated_count} credits!")
    print(f"✅ Auto-assigned IMDB codes: {auto_assigned_imdb_count}")
    print(f"✅ Auto-assigned internal codes: {auto_assigned_internal_count}")
    print(f"⚠️  Manual review required: {manual_required_count}")
    print("🎯 Now ALL credits have been processed with the corrected logic!")

if __name__ == "__main__":
    complete_imdb_fix() 