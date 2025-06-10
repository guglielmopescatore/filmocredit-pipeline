import os
from pathlib import Path
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

LOG_FILE_PATH = PROJECT_ROOT / 'filmocredit_pipeline.log'

# Path for user-defined OCR stopwords
# Path for user-defined OCR stopwords
OCR_USER_STOPWORDS_PATH: Path = PROJECT_ROOT / 'user_ocr_stopwords.txt'
DEFAULT_OCR_USER_STOPWORDS: list[str] = ["RAI", "BBC", "HBO"]

DEFAULT_OCR_ENGINE = "paddleocr"
SUPPORTED_OCR_ENGINES = ["paddleocr"] # , "easyocr"]

EASYOCR_LANG_MAP = {
    'it': ['it'],
    'en': ['en'],
    'ch': ['ch_sim', 'en'], 
}

PADDLEOCR_LANG_MAP = {
    'it': 'it',
    'en': 'en',
    'ch': 'ch',
}

CONTENT_SCENE_DETECTOR_THRESHOLD = 10.0 
THRESH_SCENE_DETECTOR_THRESHOLD = 5 
SCENE_MIN_LENGTH_FRAMES = 10

INITIAL_OCR_LANGUAGES: list[str] = ['it', 'en']
INITIAL_FRAME_SAMPLE_POINTS: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MIN_OCR_CONFIDENCE = 0.75 
MIN_OCR_TEXT_LENGTH = 4   

# Frame analysis parameters
SCROLL_FRAME_HEIGHT_RATIO = 0.9 
MIN_ABS_SCROLL_FLOW_THRESHOLD = 0.5 
OPTICAL_FLOW_PARAMS = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
HASH_SIZE = 16 
HASH_DIFFERENCE_THRESHOLD = 0
FADE_FRAME_THRESHOLD = 20.0      
FADE_FRAME_CONTRAST_THRESHOLD = 10.0 
HASH_SAMPLE_INTERVAL_SECONDS = 0.5 

FUZZY_TEXT_SIMILARITY_THRESHOLD = 60
# Dynamic threshold scaling for long texts
FUZZY_THRESHOLD_BASE = 60           # Base threshold for short texts
FUZZY_THRESHOLD_SCALE_START = 200   # Text length where scaling starts
FUZZY_THRESHOLD_SCALE_RATE = 0.02   # Increase per character after scale start (doubled)
FUZZY_THRESHOLD_MAX = 85            # Maximum threshold cap
OCR_TIMEOUT_SECONDS = 3

# Scene-based processing configuration - analyze first N and last N scenes
DEFAULT_START_SCENES_COUNT = 100  # Analyze first 100 scenes (opening credits)
DEFAULT_END_SCENES_COUNT = 100    # Analyze last 100 scenes (closing credits)

DEFAULT_VLM_MAX_NEW_TOKENS = 8192

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
    {{ "role_detail": "Specific Role/Title/Character/text that appears with the name or null", "name": "Name As Written", "role_group": "CATEGORY" }}

Example Output Format:
[
{{"role_detail": "Director", "name": "Jane Director", "role_group": "Directors"}},
{{"role_detail": "The Hero", "name": "John Actor", "role_group": "Cast"}},
{{"role_detail": "Villain", "name": "Actor One", "role_group": "Cast"}},
{{"role_detail": "e con", "name": "Jane Doe", "role_group": "Cast"}},
{{"role_detail": "Aiuti segreteria di produzione", "name": "Fabio Lucilli", "role_group": "Production Management"}},
{{"role_detail": "Musiche originali", "name": "Composer Name", "role_group": "Composers"}},
{{"role_detail": "Production Designer", "name": "Designer Name", "role_group": "Production Design"}},
{{"role_detail": "Ringraziamenti speciali", "name": "Special Thanks Org", "role_group": "Thanks"}},
{{"role_detail": "Da cosa è accompagnato il testo", "name": "Just A Name", "role_group": "Unknown"}},
{{"role_detail": "Direttore di seconda unità", "name": "Assistant Name", "role_group": "Second Unit Directors or Assistant Directors"}}
]

Field Definitions:

    role_detail: The precise textual role, character, or qualifier (e.g., "Sound Mixer", "Anna Rossi", "featuring", "e con", "e di", "presenta"). If a name appears under a category heading without an individual specific detail, set to null.
    name: The exact person or company name.    role_group: Choose only from the following predefined categories. Assign carefully based on the visible role detail or category heading in the credits:
        - **Directors**: regista, regia, directed by, co-director.
        - **Writers**: sceneggiatura, scritto da, written by, story by.
        - **Cast**: attori, interpreti, character names, "con", "e con", "e di", or names without technical roles.
        - **Producers**: producer, executive producer, coproducer, line producer.
        - **Composers**: For authors of original music or songs (e.g., "Musiche originali", "Original Music by", "Score by", "Songs by"). Not used for music technicians or coordinators.
        - **Music Department**: All other music-related functions (e.g., "Orchestrazione", "Coordinamento musicale", "Music Supervisor", "Score Mixer", "Arrangiamenti").
        - **Cinematographers**: direttore della fotografia, director of photography, camera director.
        - **Editors**: Only main editors.
        - **Casting Credits**: casting director, selezione del cast.
        - **Production Design**: scenografia, production designer, designed by.
        - **Art Directors**: art director, direzione artistica.
        - **Set Decorators**: arredo di scena, set decoration.
        - **Costume Design**: costumi, costume designer.
        - **Makeup Department**: trucco, hair & makeup, parrucchiere.
        - **Production Management**: direttore di produzione, production manager, coordinamento produzione.
        - **Second Unit Directors or Assistant Directors**: regia seconda unità, aiuto regista.
        - **Art Department**: property master, grafica, costruzione scenica, oggetti di scena.
        - **Sound Department**: sound designer, fonico, recording mixer, dialog editing.
        - **Special Effects**: effetti speciali pratici, SFX, make-up FX.
        - **Visual Effects**: VFX, effetti visivi, digital FX, CGI.
        - **Stunts**: stunt, controfigure, stunt coordinator.
        - **Camera and Electrical Department**: operatore camera, focus puller, gaffer, elettricisti.
        - **Animation Department**: animazione, character animation, layout artist.
        - **Casting Credits**: assistenti al casting, casting assistant.
        - **Costume and Wardrobe Department**: guardaroba, assistente costumi, dresser.
        - **Editorial Department**: assistenti al montaggio, editorial assistant. The editorial department includes all film/video editing functions other than the main editor.
        - **Location Management**: location manager, location scout, gestione location.
        - **Script and Continuity Department**: supervisione script, continuity, segretaria di edizione.
        - **Transportation Department**: trasporti, autisti, coordinatore trasporti.
        - **Additional Crew**: any identifiable crew member whose role doesn't fall into the above categories (e.g., "assistente generico", "aiuto produzione", "PA").
        - **Thanks**: acknowledgments, ringraziamenti, special thanks, dediche.
        - **Miscellaneous Companies**: Any company or organization mentioned other than distributors, production and post-production companies, sales agents, and special/visual effects, sound and video editing and every company 'audiovisual related' that can fit in the other categories (e.g., miscellaneous companies can be "Catering", "Cibo", "Noleggio Attrezzature", "Sponsor", "Noleggio attrezzature" etc. etc.).
        - **Unknown**: Use only if the category of role_group cannot be inferred from the visual context, and no heading or role_detail is visible.

IMPORTANT: If a role cannot be properly categorized into any of the above role groups, it should be flagged for manual review and verification.

Consecutive names under the same role heading share the same role_group, so if in the previous image the last name mentioned was under "cast" and no new role is present in the current image, probably the same role_group should be used.
Note that also companies name can fall in any of the above categories, so if a company is mentioned under "Production Design" or "Visual Effects" it should be categorized accordingly.
Ensure output is ONLY the raw JSON list, without any additional text or formatting.
"""

def is_cuda_available() -> bool:
    """Check if CUDA is available, with lazy import to avoid DLL conflicts."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

DB_TABLE_EPISODES = "episodes"
DB_TABLE_CREDITS = "credits"

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
    OCR_USER_STOPWORDS_PATH: Path = OCR_USER_STOPWORDS_PATH

@dataclass(frozen=True)
class OCRConfig:
    DEFAULT_ENGINE: str = DEFAULT_OCR_ENGINE
    SUPPORTED_ENGINES: list[str] = field(default_factory=lambda: SUPPORTED_OCR_ENGINES.copy())
    DEFAULT_LANGUAGES: list[str] = field(default_factory=lambda: INITIAL_OCR_LANGUAGES.copy())
    MIN_CONFIDENCE: float = MIN_OCR_CONFIDENCE
    MIN_TEXT_LENGTH: int = MIN_OCR_TEXT_LENGTH
    TIMEOUT_SECONDS: int = OCR_TIMEOUT_SECONDS

@dataclass(frozen=True)
class SceneDetectionConfig:
    CONTENT_THRESHOLD: float = CONTENT_SCENE_DETECTOR_THRESHOLD
    THRESHOLD: float = THRESH_SCENE_DETECTOR_THRESHOLD
    MIN_LENGTH_FRAMES: int = SCENE_MIN_LENGTH_FRAMES
    INITIAL_FRAME_SAMPLE_POINTS: list[float] = field(default_factory=lambda: INITIAL_FRAME_SAMPLE_POINTS.copy())

@dataclass(frozen=True)
class VLMConfig:
    """Configuration for Vision-Language Model (VLM) processing."""
    BASE_PROMPT_TEMPLATE: str = BASE_PROMPT_TEMPLATE
    DEFAULT_MAX_NEW_TOKENS: int = DEFAULT_VLM_MAX_NEW_TOKENS

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

# Contrast-based frame selection
MIN_CONTRAST_IMPROVEMENT_THRESHOLD: float = 5.0  # Minimum contrast difference to justify replacement
CONTRAST_CALCULATION_METHOD: str = "stddev"  # "stddev" or "laplacian"

# Valid role groups for validation
VALID_ROLE_GROUPS = {
    "Directors",
    "Writers", 
    "Cast",
    "Producers",
    "Composers",
    "Music Department",
    "Cinematographers",
    "Editors",
    "Casting",
    "Production Design",
    "Art Directors",
    "Set Decorators",
    "Costume Design",
    "Makeup Department",
    "Production Management",
    "Second Unit Directors or Assistant Directors",
    "Art Department",
    "Sound Department",
    "Special Effects",
    "Visual Effects",
    "Stunts",
    "Camera and Electrical Department",
    "Animation Department",
    "Casting Credits",
    "Costume and Wardrobe Department",
    "Editorial Department",
    "Location Management",
    "Script and Continuity Department",    
    "Transportation Department",
    "Additional Crew",
    "Thanks",
    "Miscellaneous Companies",
    "Unknown"
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
        return False, f"FLAGGED FOR REVIEW: '{role_group}' is not in the predefined role groups list. Please verify and categorize manually."

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