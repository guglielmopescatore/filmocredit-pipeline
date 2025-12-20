import sqlite3
from scripts_v3.role_detail_mapping import find_role_group_for_detail, normalize_role_detail

conn = sqlite3.connect('db/tvcredits_v3.db')
cursor = conn.cursor()

# Get all credits with role_detail and current role_group
cursor.execute('SELECT id, role_detail, role_group_normalized, is_person FROM credits WHERE role_detail IS NOT NULL AND role_detail != ""')
rows = cursor.fetchall()

corrections_needed = []
for row in rows:
    id_, role_detail, current_group, is_person = row
    
    # Skip if not a person or if Cast
    if is_person == 0 or current_group == 'Cast':
        continue
    
    normalized = normalize_role_detail(role_detail)
    if not normalized:
        continue
        
    correct_group = find_role_group_for_detail(normalized)
    
    if correct_group and correct_group != current_group:
        corrections_needed.append((id_, role_detail, current_group, correct_group))

print(f'Total records: {len(rows)}')
print(f'Corrections needed: {len(corrections_needed)}')
print()
print('Sample corrections:')
for c in corrections_needed[:30]:
    print(f'  ID {c[0]}: "{c[1]}" -> {c[2]} => {c[3]}')
