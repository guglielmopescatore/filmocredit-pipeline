from pathlib import Path
from typing_extensions import Final

from dotenv import load_dotenv

load_dotenv()

try:
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path('.').resolve()

DATA_DIR: Path = PROJECT_ROOT / 'data'
RAW_VIDEO_DIR: Path = DATA_DIR / 'raw'
EPISODES_BASE_DIR: Path = DATA_DIR / 'episodes'
DB_DIR: Path = PROJECT_ROOT / 'db'
DB_PATH: Path = DB_DIR / 'tvcredits_v3.db'
ROLE_MAP_PATH: Path = PROJECT_ROOT / 'scripts_v3' / 'mapping_ruoli.json'

# IMDB Database Configuration
IMDB_PARQUET_PATH: Path = PROJECT_ROOT / 'db' / 'normalized_names.parquet'
# Legacy TSV path for backward compatibility
IMDB_TSV_PATH: Path = PROJECT_ROOT / 'db' / 'name.basics.tsv'


# Database
DB_TABLE_CREDITS: Final[str] = "credits"
DB_TABLE_EPISODES: Final[str] = "episodes"

LOG_FILE_PATH = PROJECT_ROOT / 'filmocredit_pipeline.log'

# Path for user-defined OCR stopwords
OCR_USER_STOPWORDS_PATH: Path = PROJECT_ROOT / 'user_ocr_stopwords.txt'
DEFAULT_OCR_USER_STOPWORDS: list[str] = ["RAI", "BBC", "HBO"]
DEFAULT_OCR_ENGINE: Final[str] = "paddleocr"

SUPPORTED_OCR_ENGINES = ["paddleocr"] 

PADDLEOCR_LANG_MAP = {
    'it': 'it',
    'en': 'en',
    'ch': 'ch',
}

INITIAL_OCR_LANGUAGES: list[str] = ['it', 'en']

BASE_PROMPT_TEMPLATE = """
Objective: Extract new film credits from the current image of rolling credits, comparing against credits from the immediately preceding image.

Input:
    Current image.
    previous_credits_json: New credits identified in the frame before this one:

    {previous_credits_json}

Instructions:

    Parse all visible textual credits (role-name pairs) in the current image.
    CRITICAL: If no new credits are identified (or none are found at all), output an empty list: []. Do not fabricate any information.

Output:
    Return a raw JSON list, where each object represents a newly identified credit:
    {{ "role_detail": "Specific Role/Title/Character/text that appears with the name or null", "name": "Name As Written", "role_group": "CATEGORY", "is_person": true/false }}

Example Output Format:
[
{{"role_detail": "Director", "name": "Jane Director", "role_group": "Directors", "is_person": true}},
{{"role_detail": "The Hero", "name": "John Actor", "role_group": "Cast", "is_person": true}},
{{"role_detail": "Villain", "name": "Actor One", "role_group": "Cast", "is_person": true}},
{{"role_detail": "e con", "name": "Jane Doe", "role_group": "Cast", "is_person": true}},
{{"role_detail": "Aiuti segreteria di produzione", "name": "Fabio Lucilli", "role_group": "Production Managers", "is_person": true}},
{{"role_detail": "Musiche originali", "name": "Composer Name", "role_group": "Composers", "is_person": true}},
{{"role_detail": "Production Designer", "name": "Designer Name", "role_group": "Production Designers", "is_person": true}},
{{"role_detail": "Ringraziamenti speciali", "name": "Warner Bros. Pictures", "role_group": "Thanks", "is_person": false}},
{{"role_detail": "Catering", "name": "ABC Catering Services", "role_group": "Miscellaneous Companies", "is_person": false}},
{{"role_detail": "Da cosa è accompagnato il testo", "name": "Just A Name", "role_group": "Additional Crew", "is_person": true}},
{{"role_detail": "Direttore di seconda unità", "name": "Assistant Name", "role_group": "Second Unit Directors or Assistant Directors", "is_person": true}}
]

Field Definitions:

    role_detail: The precise textual role, character, or qualifier (e.g., "Sound Mixer", "Anna Rossi", "featuring", "e con", "e di", "presenta"). If a name appears under a category heading without an individual specific detail, set to null.
    name: The exact person or company name.
    is_person: Set to true if the name refers to an individual person, false if it refers to a company, organization, or corporate entity.

role_group: Choose only from the following predefined categories based on IMDb's classification system. Assign carefully based on the visible role detail or category heading in the credits:

**CAST AND CREW CATEGORIES:**
- **Cast**: attori, interpreti, character names, "con", "e con", "e di", or names without technical roles.
- **Directors**: regista, regia, directed by, co-director.
- **Writers**: sceneggiatura, scritto da, written by, story by.
- **Producers**: producer, executive producer, coproducer, line producer.
- **Cinematographers**: direttore della fotografia, director of photography, camera director.
- **Editors**: Only main editors.
- **Composers**: For authors of original music or songs (e.g., "Musiche originali", "Original Music by", "Score by", "Songs by"). Not used for music technicians or coordinators.
- **Production Designers**: scenografia, production designer, designed by.
- **Art Directors**: art director, direzione artistica.
- **Set Decorators**: arredo di scena, set decoration.
- **Costume Designers**: costumi, costume designer.
- **Makeup Department**: trucco, hair & makeup, parrucchiere.
- **Sound Department**: sound designer, fonico, recording mixer, dialog editing.
- **Visual Effects**: VFX, effetti visivi, digital FX, CGI.
- **Special Effects**: effetti speciali pratici, SFX, make-up FX.
- **Music Department**: All other music-related functions (e.g., "Orchestrazione", "Coordinamento musicale", "Music Supervisor", "Score Mixer", "Arrangiamenti").
- **Production Managers**: direttore di produzione, production manager, coordinamento produzione.
- **Location Managers**: location manager, location scout, gestione location.
- **Casting Directors**: assistenti al casting, casting assistant, casting director, selezione del cast.
- **Second Unit Directors or Assistant Directors**: regia seconda unità, aiuto regista.
- **Camera and Electrical Department**: operatore camera, focus puller, gaffer, elettricisti.
- **Art Department**: property master, grafica, costruzione scenica, oggetti di scena.
- **Animation Department**: animazione, character animation, layout artist.
- **Costume and Wardrobe Department**: guardaroba, assistente costumi, dresser.
- **Editorial Department**: assistenti al montaggio, editorial assistant. The editorial department includes all film/video editing functions other than the main editor.
- **Script and Continuity Department**: supervisione script, continuity, segretaria di edizione.
- **Transportation Department**: trasporti, autisti, coordinatore trasporti.
- **Stunts**: stunt, controfigure, stunt coordinator.
- **Thanks**: acknowledgments, ringraziamenti, special thanks, dediche.
- **Additional Crew**: any identifiable crew member whose role doesn't fall into the above categories.

**COMPANY CATEGORIES:**
- **Production Companies**: All financing entities, including those noted as "in association with" or "participating".
- **Distributors**: Companies that distribute the film to theaters, streaming, or other venues.
- **Sales Representatives / ISA**: International sales agents or producers' reps that sell distribution rights.
- **Special Effects Companies**: Companies providing special effects services.
- **Miscellaneous Companies**: Any company or organization mentioned other than the above categories (e.g., "Catering", "Cibo", "Noleggio Attrezzature", "Sponsor", etc.).

IMPORTANT NOTES:
- Production services or facilities companies belong in Miscellaneous Companies, not Production Companies.
- Visual Effects companies should be categorized as Special Effects Companies in the company section.
- Consecutive names under the same role heading share the same role_group.
- Companies mentioned under specific technical categories (like under "Visual Effects" heading) should still use the appropriate crew category, not the company category.
- Songs or music names should not be counted as entities - extract the composer(s) instead.
- The "name" field should be exclusively the denomination of the person or company, without the text that surrounds it in the credits (e.g., "featuring", "e con", "e di", "mixato da e presso il" etc.).
- For every entry, only one name should be provided in the "name" field, even if multiple names are listed together in the credits. If multiple names are present, include them as separate entries in the output list.
- Be sure to assign to person roles only those roles that are clearly identifiable as individual people. if a name refers to a company or organization, set "is_person" to false and categorize it under the appropriate company role group.
- A name like "Pierluigi Pardo s.r.l" or "Digital Work Alessio Marinelli" should be categorized as a company, with "is_person" set to false. This is critical to avoid misclassifying companies as individuals.
- the name field should contain ONLY the name of the person or company as it appears in the credits, without any additional text or qualifiers
- if a role cannot be properly categorized into any of the above role groups or you are in doubt, it should be flagged for manual review using "Unknown".

VERY IMPORTANT: Companies can fall only into the "Production Companies", "Distributors", "Sales Representatives / ISA", "Special Effects Companies", or "Miscellaneous Companies" categories. So, if is_person is false, the role_group must be one of these company categories. If the role does not fit any of these categories, put it in Miscellaneous Companies.

Ensure output is ONLY the raw JSON list, without any additional text or formatting.
"""




from dataclasses import dataclass, field


@dataclass(frozen=True)
class PathConfig:
    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_DIR: Path = DATA_DIR
    RAW_VIDEO_DIR: Path = RAW_VIDEO_DIR
    EPISODES_BASE_DIR: Path = EPISODES_BASE_DIR
    DB_DIR: Path = DB_DIR
    DB_PATH: Path = DB_PATH
    ROLE_MAP_PATH: Path = ROLE_MAP_PATH
    LOG_FILE_PATH: Path = LOG_FILE_PATH
    IMDB_PARQUET_PATH: Path = IMDB_PARQUET_PATH
    IMDB_TSV_PATH: Path = IMDB_TSV_PATH
    OCR_USER_STOPWORDS_PATH: Path = OCR_USER_STOPWORDS_PATH


@dataclass(frozen=True)
class OCRConfig:
    SUPPORTED_ENGINES: list[str] = field(default_factory=lambda: SUPPORTED_OCR_ENGINES.copy())
    DEFAULT_LANGUAGES: list[str] = field(default_factory=lambda: INITIAL_OCR_LANGUAGES.copy())


@dataclass(frozen=True)
class VLMConfig:
    """Configuration for Vision-Language Model (VLM) processing."""

    BASE_PROMPT_TEMPLATE: str = BASE_PROMPT_TEMPLATE


@dataclass(frozen=True)
class SceneDetectionConfig:
    pass


@dataclass(frozen=True)
class AzureConfig:
    """Environment variable keys for Azure OpenAI settings."""

    API_KEY_ENV: str = 'AZURE_OPENAI_KEY'
    API_VERSION_ENV: str = 'AZURE_OPENAI_API_VERSION'
    ENDPOINT_ENV: str = 'AZURE_OPENAI_ENDPOINT'
    DEPLOYMENT_NAME_ENV: str = 'AZURE_OPENAI_DEPLOYMENT_NAME'


# Ensure critical directories exist
for _dir in [DATA_DIR, RAW_VIDEO_DIR, EPISODES_BASE_DIR, DB_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# Valid role groups for validation
VALID_ROLE_GROUPS = {
    # Cast and Crew Categories
    "Cast",
    "Directors",
    "Writers",
    "Producers",
    "Cinematographers",
    "Editors",
    "Composers",
    "Production Designers",
    "Art Directors",
    "Set Decorators",
    "Costume Designers",
    "Makeup Department",
    "Sound Department",
    "Visual Effects",
    "Special Effects",
    "Music Department",
    "Production Managers",
    "Location Managers",
    "Casting Directors",
    "Second Unit Directors or Assistant Directors",
    "Camera and Electrical Department",
    "Art Department",
    "Animation Department",
    "Costume and Wardrobe Department",
    "Editorial Department",
    "Script and Continuity Department",
    "Transportation Department",
    "Stunts",
    "Thanks",
    "Additional Crew",
    
    # Company Categories
    "Production Companies",
    "Distributors",
    "Sales Representatives / ISA",
    "Special Effects Companies",
    "Miscellaneous Companies",
    
    # Fallback
    "Unknown",
}

# Role groups as a sorted list for UI components (selectboxes, etc.)
ROLE_GROUPS = sorted(list(VALID_ROLE_GROUPS))


def validate_role_group(role_group: str) -> tuple[bool, str]:
    """
    Validate if a role group is in the predefined list.

    Args:
        role_group: The role group to validate
          Returns:
        tuple: (is_valid, message)
            - is_valid: True if role group is valid, False otherwise
            - message: Descriptive message about the validation result
    """
    if role_group in VALID_ROLE_GROUPS:
        return True, f"Valid role group: {role_group}"
    else:
        return (
            False,
            f"FLAGGED FOR REVIEW: '{role_group}' is not in the predefined role groups list. Please verify and categorize manually.",
        )


def is_valid_role_group(role_group: str) -> bool:
    """
    Simple boolean check if a role group is valid.

    Args:
        role_group: The role group to validate

    Returns:
        bool: True if role group is valid, False otherwise
    """
    return role_group in VALID_ROLE_GROUPS


def get_flagged_role_groups(credits_data: list[dict]) -> list[dict]:
    """
    Scan a list of credits and return those with invalid role groups.

    Args:
        credits_data: List of credit dictionaries with 'role_group' field

    Returns:
        List of credits that have invalid role groups and need manual review
    """
    flagged_credits = []
    for credit in credits_data:
        role_group = credit.get('role_group', '')
        is_valid, message = validate_role_group(role_group)
        if not is_valid:
            flagged_credit = credit.copy()
            flagged_credit['validation_message'] = message
            flagged_credits.append(flagged_credit)

    return flagged_credits