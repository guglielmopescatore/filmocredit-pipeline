from pathlib import Path
from typing_extensions import Final
from . import constants
from dotenv import load_dotenv

load_dotenv()

try:
    import sys
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running as PyInstaller executable - everything goes in _MEIPASS (_internal)
        PROJECT_ROOT: Path = Path(sys._MEIPASS)
    else:
        # Running as normal Python script
        PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path('.').resolve()

DATA_DIR: Path = PROJECT_ROOT / 'data'
RAW_VIDEO_DIR: Path = DATA_DIR / 'raw'
EPISODES_BASE_DIR: Path = DATA_DIR / 'episodes'
DB_DIR: Path = PROJECT_ROOT / 'db'
DB_PATH: Path = DB_DIR / 'tvcredits_v3.db'

# IMDB Database Configuration - always in the same place as other files
IMDB_PARQUET_PATH: Path = PROJECT_ROOT / 'db' / 'normalized_names.parquet'
IMDB_TSV_PATH: Path = PROJECT_ROOT / 'db' / 'name.basics.tsv'


# Database
DB_TABLE_CREDITS: Final[str] = "credits"
DB_TABLE_EPISODES: Final[str] = "episodes"

LOG_FILE_PATH = PROJECT_ROOT / 'filmocredit_pipeline.log'

# Path for user-defined OCR stopwords - always in the same place as other files
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

### BASE PROMPT NEW
BASE_PROMPT_TEMPLATE = """
Objective: Extract new film credits from the current image of rolling credits, comparing against credits from the immediately preceding image.

Input:
Current image.
previous_credits_json: New credits identified in the frame before this one:

```
{previous_credits_json}
```

Instructions:

```
Parse all visible textual credits (role-name pairs) in the current image. Even if they are blurred or faded, attempt to identify them.
Include restoration credits (e.g., "Restored by", "Restauro a cura di"), logos with text, and technical partners.
CRITICAL: If no new credits are identified (or none are found at all), output an empty list for "credits". Do not fabricate any information.
```

Output:
Return a JSON object with the following structure:
{{
"credits": [
{{ "role_detail": "Specific Role/Title/Character/text that appears with the name or null", "name": "Name As Written", "role_group": "CATEGORY", "secondary_role_group": "FALLBACK_CATEGORY or null", "is_person": true/false }},
...
],
"explanation": "Brief explanation if credits list is empty, otherwise null"
}}

```
The "credits" field must be a list of objects where each object represents a newly identified credit.
The "explanation" field must be a string explaining why the list is empty (e.g., "No text visible", "Only logos", "Same credits as previous frame"). If credits are found, set "explanation" to null.
```

Example Output Format:
{{
"credits": [
{{"role_detail": "Director", "name": "Jane Director", "role_group": "Directors", "secondary_role_group": null, "is_person": true}},
{{"role_detail": "The Hero", "name": "John Actor, Jr.", "role_group": "Cast", "secondary_role_group": null, "is_person": true}},
{{"role_detail": "Villain", "name": "Actor One", "role_group": "Cast", "secondary_role_group": null, "is_person": true}},
{{"role_detail": "e con", "name": "Jane Doe", "role_group": "Cast", "secondary_role_group": null, "is_person": true}},
{{"role_detail": "Music Licensing", "name": "John Lawyer", "role_group": "Legal Department", "secondary_role_group": "Music Department", "is_person": true}},
{{"role_detail": "Aiuti segreteria di produzione", "name": "Fabio Lucilli", "role_group": "Production Department", "secondary_role_group": null, "is_person": true}},
{{"role_detail": "Musiche originali", "name": "Composer Name", "role_group": "Composers", "secondary_role_group": null, "is_person": true}},
{{"role_detail": "Production Designer", "name": "Designer Name", "role_group": "Production Designers", "secondary_role_group": null, "is_person": true}},
{{"role_detail": "Ringraziamenti speciali", "name": "Warner Bros. Pictures", "role_group": "Thanks", "secondary_role_group": null, "is_person": false}},
{{"role_detail": "Catering", "name": "ABC Catering Services", "role_group": "Miscellaneous Companies", "secondary_role_group": null, "is_person": false}},
{{"role_detail": "Restaurato da", "name": "Cineteca di Bologna", "role_group": "Miscellaneous Companies", "secondary_role_group": null, "is_person": false}},
{{"role_detail": "Sound Design Coordinator", "name": "Just A Name", "role_group": "Sound Department", "secondary_role_group": "Production Department", "is_person": true}},
{{"role_detail": "Direttore di seconda unità", "name": "Assistant Name", "role_group": "Second Unit Directors or Assistant Directors", "secondary_role_group": null, "is_person": true}}
],
"explanation": null
}}

OR (if no credits found):

{{
"credits": [],
"explanation": "The screen contains only the production company logo and no textual credits."
}}

Field Definitions:

```
role_detail: The precise textual role, character, or qualifier (e.g., "Sound Mixer", "Anna Rossi", "featuring", "e con", "e di", "presenta"). If a name appears under a category heading without an individual specific detail, set to null.
name: The exact person or company name.
is_person: Set to true if the name refers to an individual person, false if it refers to a company, organization, or corporate entity.
secondary_role_group: A fallback role group for ambiguous roles that could belong to multiple departments. Use this when a role clearly spans two categories. Examples:
  - "Music Licensing" → role_group: "Legal Department", secondary_role_group: "Music Department"
  - "VFX Coordinator" → role_group: "Visual Effects", secondary_role_group: "Production Department"
  - "Sound Design Coordinator" → role_group: "Sound Department", secondary_role_group: "Production Department"
  - "Script Clearance" → role_group: "Legal Department", secondary_role_group: "Script and Continuity Department"
  Set to null when the role clearly belongs to only one category. This field IS NOT for strings such as "Produced and Directed by", which should be clearly two separate credits for the same name with two different role_groups.
```

role_group: Choose only from the following predefined categories based on IMDb's classification system and the detailed descriptions below. Assign carefully based on the visible role detail or category heading in the credits:

"role_groups_definitions": {{
    "Cast": {{
      "description": "Original on-screen performers and narrators.",
      "include": [
        "Actors and performers credited as cast.",
        "Usually you can find them without headings but with nothing or just the name of the character near to them.",
        "Original voices for objects/animals ('voice of', 'voci di' in original language version).",
        "Narrators ('narrated by').",
        "Names without technical roles (e.g. 'con', 'e con', 'e di', 'han intervenido') when they clearly refer to original cast."
      ],
      "exclude": [
        "All dubbing/localized voice casts (→ 'Voice Actors - Dubbing')."
      ]
    }},

    "Sound Department": {{
      "description": "All non-dubbing sound roles except musical recording/mixing.",
      "include": [
        "sound designer, sound editor, supervising sound editor",
        "production sound mixer, re-recording mixer, recording mixer",
        "mastering ONLY when clearly sound-related",
        "boom operator, sound utility",
        "dialogue editing, ADR recording, foley recording and editing",
        "audio description engineer",
        "assistente di studio di registrazione, tecnico del suono, ingegnere del suono",
        "in french 'Son' (sound) indicates sound department roles"
      ],
      "exclude": [
        "ADR performers, loop group artists, audio description voice actors (→ 'Additional Crew')",
        "music editing/recording/mixing (→ 'Music Department')",
        "dubbing-specific sound engineers/technicians/recordist (→ 'Dubbing_Sound')",
        "when 'mastering' is found in a non-sound context is usually a video related role (→ 'Editorial Department')"
      ]
    }},

    "Dubbing_Sound": {{
      "description": "Sound technicians/engineers specifically for dubbing/localized versions.",
      "include": [
        "dubbing sound engineer, dubbing mixer, dubbing recordist when clearly tied to dubbing workflows.",
        "every sound role in credits blocks about ADR/dubbing/localized versions."
      ]
    }},

    "Voice Actors - Dubbing": {{
      "description": "Voice actors who dub localized versions of the original performance.",
      "include": [
        "\"voce italiana di\", \"voce francese di\", \"Spanish voice of\" etc.",
        "credits that clearly refer to dubbed/translated dialogue replacing the original."
      ],
      "exclude": [
        "ADR performers, loop group artists, audio description voice actors (→ 'Additional Crew')",
        "dubbing engineers and audio description engineers (→ 'Sound Department')"
      ]
    }},

    "Dubbing": {{
      "description": "Non-acting, non-sound dubbing roles (coordination, voice casting, translation, supervision for specific language versions).",
      "include": [
        "dubbing casting, dubbing coordinator, dubbing production supervisor",
        "you can also find 'casting diretto' or 'casting direção' related to dubbing and it go here, not in 'Directors' or 'Casting Directors'",
        "localization and translation roles for dubbed versions (e.g. 'Adaptation by', 'Traduzione di' for language versions)",
        "ADR coordinator when tied to a dubbed/localized version",
        "project managers, editors, directors, coordinators, writers of adaptations for specific versions (e.g. Italian, Brazilian, English version pages)"
      ]
    }},

    "Directors": {{
      "description": "Primary creative directors of the production.",
      "include": [
        "regista, regia, 'directed by', 'director', co-director (with attribute if specified)",
        "supervising directors",
        "TV series directors credited on specific episodes",
        "only when person has a key creative/directorial role"
      ]
    }},

    "Second Unit Directors or Assistant Directors": {{
      "description": "Second unit directing and assistant directing roles.",
      "include": [
        "regia seconda unità, 2nd Unit Director, 2nd Unit Coordinator",
        "aiuto regista, assistant director (1st AD, 2nd AD, 3rd AD)",
        "DGA trainee, AD PA when clearly tied to directing department",
        "set coordination/management specifically linked to second-unit directing",
        "AOSM (Organizzatore scene di massa)"
      ],
      "exclude": [
        "generic runner or PA not clearly tied to directing (→ 'Production Department')",
        "assistant to director (personal) (→ 'Production Department')",
        "supervising directors (→ 'Directors')"
      ]
    }},

    "Writers": {{
      "description": "Writing and story creation roles.",
      "include": [
        "sceneggiatura, soggetto, 'written by', 'story by', 'screenplay by', 'teleplay by'",
        "'creator' / 'created by' (TV series creators)",
        "'screen story', 'television story'",
        "source material authors (novel, book, play, article, characters etc.)",
        "staff writers, story editors, story coordinators, story supervisors, dramaturg/dramaturge"
      ],
      "exclude": [
        "script doctors / script consultants (→ 'Script and Continuity Department')",
        "adaptation/translation for localization ('Adaptation by', 'Traduzione di' for language versions) (→ 'Dubbing')"
        "'written by', 'escrito por' etc. etc. related to songs, lyrics, music",
        "Every person that writes something THAT IS NOT the screenplay or story of the movie should not go here."
      ]
    }},

    "Producers": {{
      "description": "Creative/financing producer roles.",
      "include": [
        "producer, co-producer, associate producer",
        "executive producer, co-executive producer, assistant producer (NOT assistant to producer taht go to 'Production Department')",
        "line producer when clearly a creative/producer credit rather than pure management",
        "\"produced by\", \"developed for television by\"",
        "producer (p.g.a.)",
        "commissioning editors and series editors when functioning as producing roles",
        "assistant producer (not 'assistant to producer')",
        "every Executive belongs here unless clearly a pure financial role",
        "executive in charge of production"
      ],
      "exclude": [
        "production manager, unit production manager, production supervisor, unit manager, stage manager, production director (→ 'Production Managers')",
        "company-only credits like 'Produzione Babe Films' (company → 'Miscellaneous Companies'; individual admin staff → 'Additional Crew')"
      ]
    }},

    "Production Department": {{
      "description": "General production office and logistics support (non-head roles) inside the movie production.",
      "include": [
        "assistant production manager when not the main PM",
        "Assistant Unit Managers",
        "production assistants (PAs) and executive assistants",
        "production runner, RUNNERS (also for props/arts/set dec), production secretary",
        "assistant to producer, assistant to director",
        "production coordinator, production office staff, production support staff",
        "production consultant",
        "public space/filming/location permits and similar minor permit roles",
        "department PAs should go to the specific department instead where clearly indicated",
      ],
      "exclude": [
        "production manager, unit production manager, production supervisor, post-production supervisor, unit manager, executive in charge of production, production director (→ 'Production Managers')",
        "casting roles",
        "external company credits such as 'Produzione Babe Films' because those are people external to the production (→ 'Additional Crew')",
        "interns not clearly assigned to production (→ 'Additional Crew')",
        "personal assistants to individuals such as 'Assistant to David Lynch' (→ 'Additional Crew')"
      ]
    }},

    "Cinematographers": {{
      "description": "Directors of photography (DoP/main cinematographers).",
      "include": [
        "direttore della fotografia, director of photography, 'photography by'",
        "main cinematographer(s)",
        "second unit/location cinematographers when credited with DoP/cinematographer title and leading photography for that unit (such as '2nd Unit Director of Photography' or '2ND UNIT DOP'S')"
      ]
    }},

    "Editors": {{
      "description": "Main picture editors.",
      "include": [
        "editor, film editor, picture editor",
        "edizione / montaggio when clearly referring to the main editor(s)"
      ],
      "exclude": [
        "assistant editors, additional editors, online editors, etc. (→ 'Editorial Department')"
      ]
    }},

    "Composers": {{
      "description": "Composers of the main score and original music.",
      "include": [
        "\"Music by\", \"Score by\", \"Original score composed by\"",
        "\"Musiche originali\", \"Musica originale\"",
        "authors of original songs and lyrics when clearly credited as writers (e.g. 'Songs written by', 'Lyrics by')",
        "Also co-composers when clearly indicated (such as in 'Score Co-Produced by')"
      ],
      "exclude": [
        "music supervisors, orchestrators, arrangers, music editors, performers etc. (→ 'Music Department')"
      ]
    }},

    "Production Designers": {{
      "description": "Primary production designers only.",
      "include": [
        "scenografia, 'production designer', 'designed by' when referring to the overall production design",
        "co-production designer when clearly main"
      ],
      "exclude": [
        "assistant production designers, second-unit production designers, location production designers (→ 'Art Department')",
        "set decorators (→ 'Set Decorators')"
      ]
    }},

    "Art Directors": {{
      "description": "Main art directors and supervising art directors.",
      "include": [
        "\"art director\", 'direzione artistica' for main art directors",
        "supervising art director"
      ],
      "exclude": [
        "assistant art directors, second unit/location art directors (→ 'Art Department')"
      ]
    }},

    "Set Decorators": {{
      "description": "Main unit set decorators only.",
      "include": [
        "set decorator, chief set decorator",
        "head of set decoration for the main unit", "Lead Man"
      ],
      "exclude": [
        "buyers, set dressers, assistant set decorators, daily set dressers, second-unit or location set decorators (→ 'Art Department')"
      ]
    }},

    "Costume Designers": {{
      "description": "Primary costume designer credits.",
      "include": [
        "\"Costume Designer\", 'costumi', 'gowns', 'wardrobe by' when clearly design-level",
        "co-costume designer (with attribute where present)",
        "'Vestuario', 'Costumière', 'Costumista' if not part of a block of 'helpers' or 'assistants'"
      ],
      "exclude": [
        "wardrobe, wardrobe assistants, dressers (→ 'Costume and Wardrobe Department')",
        "assistant costume designers, second unit costume designers (→ 'Costume and Wardrobe Department')",
        "credits clearly for wardrobe management rather than design"
      ]
    }},

    "Makeup Department": {{
      "description": "All makeup, hair, and wig roles related to character appearance.",
      "include": [
        "makeup, trucco, make-up artist",
        "hair stylist, parrucchiere, hair designer",
        "makeup designer, key makeup artist",
        "wig maker, wig designer, wig technician",
        "prosthetic makeup application and character prosthetics",
        "dental prosthetics, teeth",
      ],
      "exclude": [
        "prosthetics/mechanical rigs not applied as character makeup (→ 'Special Effects')",
        "costume-related fabrication (→ 'Costume and Wardrobe Department')"
      ]
    }},

    "Visual Effects": {{
      "description": "Digital/optical post-production visual effects.",
      "include": [
        "VFX supervisor, VFX producer",
        "compositor, matte painter, CG artist, matchmover",
        "simulation/FX artist, lighting TD, roto/paint, tracking",
        "digital cleanup, digital restoration, DI cleanup when under VFX",
        "miniature/model work credited under VFX",
        "animators in live-action VFX or effects sequences",
        "titles/motion graphics in a digital post-production/VFX context",
        "Research and Development (R&D) roles specifically for VFX technology"
      ],
      "exclude": [
        "practical/on-set effects (→ 'Special Effects')",
        "animation roles for general animation in animated films (→ 'Animation Department')"
      ]
    }},

    "Special Effects": {{
      "description": "Practical/on-set physical effects.",
      "include": [
        "special effects supervisor, SFX technician, FX producer",
        "pyrotechnics, explosions, squibs",
        "atmospheric effects (rain, wind machines, smoke)",
        "mechanical rigs, animatronics used live on set",
        "practical creature effects and model makers when credited under Special Effects",
        "sculptors, moulders, makeup and Prosthetic Technicians if in a section of special effects",
        "FX photography, optical effects when clearly practical/on-set",
        "Key artists when clearly part of special effects team",
        "SFX project coordinator"
      ],
      "exclude": [
        "visual/digital effects (→ 'Visual Effects')",
        "weapon masters, armorers (→ 'Property Department')",
        "model makers credited under VFX or Art (→ follow heading)"
      ]
    }},

    "Music Department": {{
      "description": "All music-related roles other than main composer(s).",
      "include": [
        "orchestrators, arrangers, music supervisors",
        "also in other languages: orchestratori, arrangiatori, supervisores musicales, arreglistas, Conductor etc. etc.",
        "music editors, additional music, music consultants",
        "music coordinators, music contractors",
        "score mixers, music recording engineers",
        "musicians and singers credited in crew (e.g. 'singer', 'musician: violin')",
        
      ],
      "exclude": [
        "main composer(s) (→ 'Composers') of the movie/production",
        "sound roles not related to music (→ 'Sound Department')",
        "song-specific credits tied to named songs (→ 'Soundtrack')",
        "music clearances/legal roles (→ 'Legal Department')"
      ]
    }},

    "Production Managers": {{
      "description": "Production management and supervisory roles handling schedule, budget, and operations.",
      "include": [
        "production manager, unit production manager (UPM)",
        "production supervisor, post-production supervisor, ispettore di produzione",
        "unit manager, production director",
        "production administrator", 
        "'Organizzazione' in italian credits when clearly production management",
        
      ],
      "exclude": [
        "assistants, coordinators, and secretaries under PMs (→ 'Production Department' or 'Additional Crew')",
        "finance/accounting roles (→ 'Production Finance and Accounting' if used; otherwise 'Additional Crew')"
      ]
    }},
    
    "Production Finance and Accounting": {{
      "description": "Financial and accounting roles within production management.",
        "include": [
            "production accountant, controller, cashier",
            "financial controller, cost report manager",
        ]
    }},

    "Location Managers": {{
      "description": "Location management and logistics for filming locations.",
      "include": [
        "location manager, assistant/associate location manager",
        "location scout, location researcher",
        "location coordinator, location assistant",
        "permitting and logistics specifically tied to locations",
        "Régisseurs adjoints in francais are part of location management"
      ],
      "exclude": [
        "location sound (→ 'Sound Department')",
        "location casting (→ 'Casting Department')",
        "studio teachers (→ 'Additional Crew')",
        "generic production managers (→ 'Production Managers')"
      ]
    }},

    "Casting Department": {{
      "description": "All non-head casting roles.",
      "include": [
        "casting assistant, casting associate, casting coordinator",
        "extras casting, regional casting ('casting: Canada', 'casting: Toronto')",
        "ADR casting",
        "\"original casting\" for non-head casting personnel"
      ],
      "exclude": [
        "voice casting for dubbed/localized versions (→ 'Dubbing')",
        "main casting director(s) (→ 'Casting Directors')"
      ]
    }},

    "Casting Directors": {{
      "description": "Main casting director(s).",
      "include": [
        "\"casting\", 'casting by' when referring to the head casting director",
        "head casting credits for the production",
        "Casting Director"
      ],
      "exclude": [
        "location/extra casting staff  (→ 'Casting Department')"
      ]
    }},
    
    {{
        "Camera and Electrical Department": {{
            "description": "All non-cinematographer non-directors camera, electrical, lighting, grip, and on-set video systems roles.",
            "include": [
                "Assistant camera, 1st AC, 2nd AC, clapper/loader, focus puller",
                "Camera operator (when not credited as cinematographer/DoP)",
                "Data wranglers, digital imaging technicians (DITs)",
                "Steadicam operator, drone operator, crane operator, jib operator",
                "Underwater camera operator, underwater photographer",
                "Digital imaging technician (DIT)",
                "Video assist operator, video playback operator, 24-frame playback, on-set video engineer",
                "Gaffer (chief lighting technician), best boy electric (assistant chief lighting technician)",
                "Electrician, generator operator, cable puller",
                "Key grip, assistant key grip, grip crew, rigging grips",
                "Photographs, dailys photographer, still photographer (on-set)",
                "video operators",
                "other languages: gruppisti, Machinistes, électriciens, fotógrafo de plató etc. etc.",
                "In products where there are a lot of 'Photography' credits, these go here unless there is a main cinematographer/DoP credited for the whole production"
            ],
            "exclude": [
                "Cinematographers / Directors of Photography (→ Cinematographers)",
                "VFX-specific camera roles (→ Visual Effects)"
            ]
        }}
    }},

    "Choreography": {{
      "description": "Dance, movement, and coordinated physical performance (non-stunt).",
      "include": [
        "choreographer, assistant choreographer, dance choreographer, dance director",
        "boxing choreographer, movement coach",
        "dance coordinator, ensemble stager"
      ],
      "exclude": [
        "fight choreographer, stunt choreographer, weaponry choreography (→ 'Stunts')",
        "intimacy choreographer (→ 'Intimacy Coordination')",
        "dancers credited in cast (→ 'Cast'); otherwise (→ 'Additional Crew')"
      ]
    }},

    "Color Department": {{
      "description": "Color correction and grading in post-production.",
      "include": [
        "colorist, color grader, color timer, DI and DI engineer",
        "dailies colorist, digital intermediate colorist, finishing colorist, HDR colorist",
        "conforming and color versioning roles tied to final color work"
      ],
      "exclude": [
        "artistic color design as part of Animation or Art (→ 'Animation Department' or 'Art Department')"
      ]
    }},

    "Costume and Wardrobe Department": {{
      "description": "Costume and wardrobe roles below the main designer level.",
      "include": [
        "assistant costume designers, second-unit costume designers",
        "wardrobe, wardrobe assistants, dressers, costumers, drapers",
        "costume design assistants, costume/wardrobe supervisors",
        "costume makers for hats/clothing unless clearly makeup-related"
      ],
      "exclude": [
        "wigs (→ 'Makeup Department')"
      ]
    }},

    "Craft Services": {{
      "description": "On-set catering and refreshments.",
      "include": [
        "caterer, catering chef, catering coordinator",
        "craft service assistant, craft server, craft services"
      ],
      "exclude": [
        "food stylist (→ 'Property Department')"
      ]
    }},

    "Art Department": {{
      "description": "All art/prop/set-construction roles not reserved for heads of department.",
      "include": [
        "assistant art director, second unit/location art director",
        "set dresser, props, scenic painters, carpenters, greens",
        "set construction, set dressing, graphic design, sign writers, Décors (french), assistant set decorator",
        "production buyer, set decoration buyer, set dressing buyer, prop buyers",
        "sculptors making set objects or decor, 'factices'"
      ],
      "exclude": [
        "primary production designer (→ 'Production Designers')",
        "primary art director and supervising art director (→ 'Art Directors')",
        "model makers in Special Effects blocks of credits (→ 'Special Effects')",
        "costume-related work (→ 'Costume and Wardrobe Department')",
        "propmakers (→ 'Property Department')",
        "When there is just one or few names with things such as 'Arredamento' or 'Ambientadora' or 'Arredatore' (not their assistants) might be the main Set Decorator and go to 'Set Decorators', not here"
      ]
    }},

    "Animation Department": {{
      "description": "Animation roles not classified under Visual Effects.",
      "include": [
        "character animator, 2D/3D animator, layout artist, digital animator",
        "storyboard artist for animated productions or animation sequences",
        "character/prop designers for animation, riggers, modelers for animation",
        "title sequence director, title designer, motion graphics artist when clearly animation/graphics work"
      ],
      "exclude": [
        "effects animators on animated films clearly part of VFX (→ 'Visual Effects')",
        "storyboard artists working for live-action shoots (→ 'Art Department')",
        "animaliers (animal handling) (→ 'Additional Crew')"
      ]
    }},

    "Editorial Department": {{
      "description": "All editing/post roles not covered by main editor, color, VFX, or sound.",
      "include": [
        "assistant editors, additional editors (support roles), on-line editors",
        "negative cutter, dailies operator, post-production coordinator",
        "editorial assistants, post-production assistants when editorial-focused",
        "video mastering, VOD mastering, data I/O, conforming when under editorial",
        "head of post-production",
        "subtitle creation and timing when clearly editorial/post",
        "in francais: assistant monteur, monteur additionnel, monteur en ligne, traitement numérique, responsable de l'argentique, Numérisation et conformation etc"
      ],
      "exclude": [
        "sound editing (→ 'Sound Department')",
        "VFX editing (→ 'Visual Effects')",
        "commissioning editors / series editors as producing roles (→ 'Producers')",
        "post-production supervisors (→ 'Production Managers')",
        "administrative/managerial roles in external post houses like 'Technical Director' (→ 'Additional Crew')"
      ]
    }},

    "Health and Safety Department": {{
      "description": "Health, safety, and compliance roles on set.",
      "include": [
        "covid supervisor, covid compliance officer",
        "health and safety advisor, hygiene officer, first aid",
        "fire safety officer, safety officer, risk assessor",
        "set medic, unit nurse, water safety coordinator"
      ],
      "exclude": [
        "animal safety/handling (→ 'Additional Crew')",
        "stunt fire/safety supervisors (→ 'Stunts')",
        "medical property master (→ 'Property Department')"
      ]
    }},

    "Intimacy Coordination": {{
      "description": "Roles overseeing scenes involving intimacy or nudity.",
      "include": [
        "intimacy coordinator, intimacy director",
        "intimacy coach, intimacy consultant",
        "intimacy choreographer"
      ],
      "exclude": [
        "PA to intimacy coordinator (→ 'Additional Crew')",
        "acting coach (→ 'Additional Crew')",
        "stunt intimacy coordinator (→ 'Stunts')"
      ]
    }},

    "Legal Department": {{
      "description": "Legal and clearance-related roles.",
      "include": [
        "attorney, legal counsel, production lawyer",
        "clearance producer, clearance supervisor, director of rights and clearances",
        "head of legal, legal executive, compliance officer, paralegal",
        "script clearance, music clearance"
      ],
      "exclude": [
        "accounting legal coordinator (→ 'Additional Crew')"
      ]
    }},

    "Publicity": {{
      "description": "Public image, press, and media relations roles.",
      "include": [
        "publicist, unit publicist",
        "public relations specialist, PR coordinator"
      ],
      "exclude": [
        "advertising manager, marketing director/executive/lead/manager",
        "poster/trailer marketing roles (→ 'Additional Crew')"
      ]
    }},

    "Puppetry": {{
      "description": "Puppet and physical character manipulation roles.",
      "include": [
        "puppeteer, assistant puppeteer, puppet operator, puppet wrangler",
        "puppet builder, puppet maker, puppet designer, puppet technician",
        "marionette designer"
      ],
      "exclude": [
        "puppet animator for animation (→ 'Animation Department')",
        "puppet costumer (→ 'Costume and Wardrobe Department')",
        "unclear puppet/props mixed roles (often → 'Additional Crew')"
      ]
    }},

    "Script and Continuity Department": {{
      "description": "Script, continuity, and script-related consulting.",
      "include": [
        "script supervisor, script coordinator",
        "continuity supervisor, continuity assistant",
        "script editor when in production context (not staff writer)",
        "script doctors, script consultants",
        "tracking script revisions, maintaining shooting script and continuity notes"
        "in italian: 'segretario di edizione (SEGR. ED.)', 'supervisore della continuità', 'assistente alla continuità', 'consulente alla sceneggiatura', 'revisore della sceneggiatura'"
      ],
      "exclude": [
        "visual/sound continuity tied to specific departments (animation, ADR, etc.) (→ relevant department)"
      ]
    }},
    
    "Property Department": {{
        "description": "Responsible for all movable objects handled by actors or used as set dressing. Sources, creates, maintains, and manages props that support storytelling.",
        "include": [
            "armorer, weapons master",
            "property master, assistant property master",
            "property assistant, props runner",
            "props builder, propsman, propmakers (very important)",
            "food stylist",
            "property department (general credit)"
        ],
        "exclude": [
            "costume maker (→ Costume Department)",
            "model maker if credited under SFX/VFX departments (→ stay in SFX/VFX)",
            "production buyer (→ Art Department)",
            "props driver (→ Transportation Department)",
            "set decoration buyer, set dressing buyer (→ Art Department)"
        ]
    }},

    "Soundtrack": {{
      "description": "Credits tied to a specific named song.",
      "include": [
        "songwriters, composers, lyricists when tied to a named song title",
        "performers, producers, arrangers, music publishers, labels for specific songs",
        "phrases like 'Written by', 'Performed by', 'Produced by', 'Arranged by', 'Courtesy of', 'Published by' in song blocks"
      ],
      "notes": [
        "Song titles go in `role_detail`, not in `name`.",
        "Each contributor must be listed as a separate JSON object.",
        "It's very important that you understand the songs blocks and you use this category for all the people working in the song."
      ],
      "example": [
        {{
            "role_detail": "My Heart Will Go On", 
            "name": "Celine Dion", 
            "role_group": "Soundtrack", 
            "secondary_role_group": null, 
            "is_person": true 
        }}
      ],
      "exclude": [
        "score or music not tied to specific song titles (→ 'Composers' / 'Music Department')",
        "sound roles not associated with a named song (→ 'Sound Department')"
      ]
    }},

    "Stunts": {{
      "description": "Stunt performances and stunt coordination.",
      "include": [
        "stunt performer, stunt doubles, stunt coordinator, pilots",
        "fight coordinators, stunt riggers, driving doubles",
        "aerial coordinators, marine/naval coordinators",
        "stunt intimacy coordinator (if clearly within stunt department)"
      ],
      "exclude": [
        "photo doubles and stand-ins (→ 'Additional Crew')"
      ]
    }},

    "Thanks": {{
      "description": "Acknowledgements, thanks, and dedications (people only).",
      "include": [
        "\"special thanks\", 'ringraziamenti', 'acknowledgements'",
        "In memoriam, dedications"
      ],
      "exclude": [
        "companies or amorphous groups (→ 'Miscellaneous Companies')",
        "Production Babies (not handled here)"
      ]
    }},

    "Transportation Department": {{
      "description": "Transportation and vehicles logistics.",
      "include": [
        "drivers, unit drivers, transport captains",
        "transportation coordinator, transportation manager",
        "shipping/vehicle logistics, picture car drivers, parking coordination",
        "all drivers also for set decoration/props transport"
      ],
      "exclude": [
        "travel and accommodation roles (→ 'Additional Crew')"
      ]
    }},

    "Additional Crew": {{
      "description": "All identifiable crew not fitting any specific category above.",
      "include": [
        "facilities, people external to the production of the movie"
        "sales roles and distributors as individuals (e.g., 'sales representative', 'distribution executive')",
        "stand-ins and photo doubles (e.g. 'stand-in', 'stand-in: [actor name]')",
        "assistant stage managers, floor managers",
        "assistants to unidentified people or departments, personal assistants",
        "translators, dialect coaches, animal trainers, religious advisors, historical consultants",
        "interns and stagiaires not clearly assigned to a specific department (try to infer department if possible)",
        "individuals working for external companies in non-producer roles (e.g. 'Produzione [Company]' named individuals who are not producers)"
      ]
    }}
  }},

  "company_groups_definitions": {{
    "Production Companies": {{
      "description": "Financing and production entities.",
      "include": [
        "\"Produced by [Company]\", 'in association with', 'production services by' when clearly production",
        "co-production partners",
        "companies 'presenting' a film where context indicates production"
      ],
      "notes": [
        "Production service providers without financing can still go here if explicitly credited as production companies; otherwise use 'Miscellaneous Companies'."
      ]
    }},

    "Distributors": {{
      "description": "Companies that distribute the film.",
      "include": [
        "theatrical distributors, TV broadcasters, streaming platforms when credited as distributors",
        "regional distribution partners (e.g. 'Distribuzione Italia', 'U.S. Distributor')"
      ]
    }},

    "Sales Representatives / ISA": {{
      "description": "International sales agents and producers' representatives.",
      "include": [
        "international/world sales, sales agent, producers' representative companies",
        "companies responsible for selling distribution rights"
      ]
    }},

    "Special Effects Companies": {{
      "description": "Companies providing Special or Visual Effects.",
      "include": [
        "VFX houses, SFX vendors",
        "digital post houses when clearly tied to VFX/SFX work"
      ]
    }},

    "Miscellaneous Companies": {{
      "description": "Any company/organization not fitting other company categories.",
      "include": [
        "catering companies, equipment rental, studios, labs, restoration facilities",
        "sponsors, brands, funding bodies, film commissions, archives",
        "'Restored by', 'Restauro a cura di', 'Color by', 'Post-production by', 'Facility services', etc."
      ]
    }}
  }},
  "extraction_rules_and_guidelines": {{
    "formatting_and_entity_extraction": [
      "Name Cleaning: The 'name' field must contain ONLY the denomination (no 'featuring', 'e con', 'mixato da', titles like 'Dr.', 'Mr.'). Keep 'Jr.'/'Sr.'.",
      "Singular Entities: Only one name per JSON object. If multiple names are listed on one line, create separate objects.",
      "Song Titles: Do not extract song/track titles as entities. Extract only composers and performers.",
      "Context Inference: Consecutive names under the same heading share the same role_group. If no role is visible, infer from previous frame.",
      "Dual Roles: If a name has two distinct roles (e.g., 'Production Executive' and 'Construction Coordinator'), create TWO separate entries with appropriate groups."
    ],
    "people_vs_companies_strict": [
      "Identification: If a name refers to a corporate entity (contains 's.r.l.', 'Inc.', 'GmbH', 'Studios' or acts as a brand), set 'is_person': false.",
      "Strict Categorization: Companies can ONLY appear in: 'Production Companies', 'Distributors', 'Sales Representatives / ISA', 'Special Effects Companies', or 'Miscellaneous Companies'.",
      "No Mixing: Never put a company in a person-based category (like 'Producers' or 'Visual Effects').",
      "Fallback: If a company doesn't fit the specific categories, use 'Miscellaneous Companies'."
    ],
    "company_specific_assignments": [
      "Production Services: Post houses, labs, rental houses go to 'Miscellaneous Companies' (not Production Companies) unless explicitly 'Produced by'.",
      "VFX Vendors: Visual Effects vendors (VFX or SFX) MUST go to 'Special Effects Companies'.",
      "Tech Headings: Companies listed under 'Visual Effects' or technical headers still go to 'Special Effects Companies' (or 'Miscellaneous'), NOT the department role."
    ],
    "role_group_logic_and_edge_cases": [
      "Songs vs Score: 'Soundtrack' is for ALL people working on songs. 'Music Department'/'Composers' is ONLY for the film's score/crew.",
      "Department Context: 'Executive Producer', 'Coordinator', 'Project Manager' listed under a specific Dept (e.g., 'Visual Effects') go to THAT Department, not 'Producers'.",
      "Interns: Interns under a specific department go to that department (e.g., Editorial Intern -> Editorial Department), not Additional Crew.",
      "Production Managers (IMDb Rule): Only HEADS (UPM, Supervisor, Line Producer) go here. Coordinators/Assistants go to 'Production Department'.",
      "Aiuto Attrezzisti: Usually 'Camera and Electrical', but 'Art Department' if tied to props/prep.",
      "Helpers/Refuerzos/Peones: Subheadings like 'Producción' under a 'Helpers' block = 'Production Department' (assistants); subheadings like "Dirección" = 'Second Unit Directors or Assistant Directors' etc. etc..",
      "VFX Context: Subheadings like 'Producción' under 'Title Design' or 'VFX' blocks = 'Visual Effects' (digital production).",
      "Unknowns: If a role cannot be categorized or is doubtful, set 'role_group': 'Unknown' for manual review."
    ],
    "previous_credits_handling": [
      "READ-ONLY: Treat 'previous_credits_json' as context only. NEVER fix or change it.",
      "No Duplicates: The current list must contain ONLY new credits. If found in previous_json, ignore and do not write them again unless there is a clear indication in the frame that it has another role!",
      "Empty Frame: If all visible credits are already in previous_json, return empty list [] with explanation 'Same credits as previous frame'."
    ]
  }}
}}
        
Ensure output is ONLY the raw JSON object, without any additional text or formatting.
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
    CONTENT_DETECTOR_THRESHOLD: float = constants.CONTENT_SCENE_DETECTOR_THRESHOLD
    THRESHOLD_DETECTOR_THRESHOLD: float = constants.THRESH_SCENE_DETECTOR_THRESHOLD
    MIN_LENGTH_FRAMES: int = constants.SCENE_MIN_LENGTH_FRAMES
    INITIAL_FRAME_SAMPLE_POINTS: list[float] = field(default_factory=lambda: constants.INITIAL_FRAME_SAMPLE_POINTS.copy())


@dataclass(frozen=True)
class AzureConfig:
    """Environment variable keys for Azure OpenAI settings."""

    API_KEY_ENV: str = 'AZURE_OPENAI_KEY'
    # Default API version if not specified in env
    DEFAULT_API_VERSION: str = '2025-03-01-preview'
    
    # GPT 4.1
    GPT4_ENDPOINT_ENV: str = 'GPT_4_1_AZURE_OPENAI_ENDPOINT'
    GPT4_DEPLOYMENT_NAME_ENV: str = 'GPT_4_1_AZURE_OPENAI_DEPLOYMENT_NAME'
    
    # GPT 5.1
    GPT5_ENDPOINT_ENV: str = 'GPT_5_1_AZURE_OPENAI_ENDPOINT'
    GPT5_DEPLOYMENT_NAME_ENV: str = 'GPT_5_1_AZURE_OPENAI_DEPLOYMENT_NAME'
    
    # Legacy/Fallback (if needed)
    ENDPOINT_ENV: str = 'AZURE_OPENAI_ENDPOINT'
    DEPLOYMENT_NAME_ENV: str = 'AZURE_OPENAI_DEPLOYMENT_NAME'
    API_VERSION_ENV: str = 'AZURE_OPENAI_API_VERSION'


# Ensure critical directories exist (only in development mode)
# In executable mode, directories are pre-created in the bundle
if not (getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')):
    # Only create directories when running as development script
    for _dir in [DATA_DIR, RAW_VIDEO_DIR, EPISODES_BASE_DIR, DB_DIR]:
        _dir.mkdir(parents=True, exist_ok=True)

# Valid role groups for validation
VALID_ROLE_GROUPS = {
    # Cast and Crew Categories
    "Cast",
    "Voice Actors - Dubbing",
    "Directors",
    "Writers",
    "Producers",
    "Production Department",
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
    "Casting Department",
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
    "Dubbing",
    "Dubbing_Sound",
    "Choreography",
    "Color Department",
    "Craft Services",
    "Health and Safety Department",
    "Intimacy Coordination",
    "Legal Department",
    "Production Finance and Accounting",
    "Property Department",
    "Publicity",
    "Puppetry",
    "Soundtrack",
    
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


# Role group to IMDB profession mapping for nconst assignment
# This mapping is used to match credits role groups with IMDB primaryProfession values
ROLE_GROUP_TO_IMDB_PROFESSION = {
    "cast": {"actor", "actress"},
    "voice actors - dubbing": {"actor", "actress", "miscellaneous"},

    "directors": {"director"},
    "writers": {"writer"},
    "producers": {"producer", "executive"},

    "production department": {"production_department", "miscellaneous"},
    "cinematographers": {"cinematographer"},
    "editors": {"editor"},
    "composers": {"composer", "soundtrack"},

    "production designers": {"production_designer"},
    "art directors": {"art_director"},
    "set decorators": {"set_decorator"},
    "costume designers": {"costume_designer"},

    "makeup department": {"make_up_department"},
    "sound department": {"sound_department"},
    "visual effects": {"visual_effects"},
    "special effects": {"special_effects"},

    "music department": {"music_department"},
    "production managers": {"production_manager"},
    "location managers": {"location_management", "miscellaneous"},

    "casting directors": {"casting_director"},
    "casting department": {"casting_department"},

    "second unit directors or assistant directors": {"assistant_director"},
    "camera and electrical department": {"camera_department", "electrical_department"},

    "art department": {"art_department"},
    "animation department": {"animation_department"},
    "costume and wardrobe department": {"costume_department"},
    "editorial department": {"editorial_department"},
    "script and continuity department": {"script_department"},
    "transportation department": {"transportation_department"},

    "stunts": {"stunts"},
    "additional crew": {"miscellaneous"},

    "dubbing": {"miscellaneous"},
    "dubbing_sound": {"sound_department", "miscellaneous"},

    "choreography": {"choreographer", "miscellaneous"},
    "color department": {"editorial_department", "miscellaneous"},
    "craft services": {"miscellaneous"},
    "health and safety department": {"miscellaneous"},
    "intimacy coordination": {"miscellaneous"},

    "legal department": {"legal", "miscellaneous"},

    "production finance and accounting": {"accountant", "miscellaneous"},

    "property department": {"miscellaneous"},

    "publicity": {"publicist", "miscellaneous"},
    
    "puppetry": {"miscellaneous","animation_department", "art_department"},

    "soundtrack": {"soundtrack", "composer", "music_department", "miscellaneous"},

    # "thanks" has no profession mapping — handled separately
}


def get_imdb_professions_for_role_group(role_group: str) -> set[str]:
    """
    Get the set of IMDB professions that correspond to a given role group.
    
    Args:
        role_group: The role group to look up
        
    Returns:
        set: Set of IMDB professions, empty set if no mapping exists
    """
    if not role_group:
        return set()
    
    # Normalize role group to lowercase for comparison
    normalized_role_group = role_group.lower()
    return ROLE_GROUP_TO_IMDB_PROFESSION.get(normalized_role_group, set())


def has_imdb_profession_mapping(role_group: str) -> bool:
    """
    Check if a role group has a direct IMDB profession mapping.
    
    Args:
        role_group: The role group to check
        
    Returns:
        bool: True if the role group maps to IMDB professions, False otherwise
    """
    return len(get_imdb_professions_for_role_group(role_group)) > 0


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