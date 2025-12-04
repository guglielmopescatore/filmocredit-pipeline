"""
Role Detail to Role Group Hard Mapping

This module provides automatic correction of LLM-assigned role_group based on role_detail keywords.
When enabled, it overrides incorrect LLM categorizations using EXACT pattern matching on role_detail text.

IMPORTANT: All patterns use ^ and $ to match the ENTIRE role_detail string exactly.
This prevents false positives like "Written & Narrated by" matching "narrated by".
"""

import re
from typing import Optional, Dict, List, Set
import logging


ROLE_DETAIL_TO_ROLE_GROUP_MAPPING: Dict[str, List[str]] = {

    # ==================== DIRECTORS ====================
    # Strictly main directors only.
    "Directors": [
        r"^director$",
        r"^directed by$",
        r"^a film by$",
        r"^un film di$",
        r"^co-?director$",
        r"^regia$",
        r"^regista$",
        r"^réalisateur$",
        r"^réalisation$",
        r"^dirección$",
        r"^dirigido por$",
        r"^mise en scène$",
    ],

    # ==================== WRITERS ====================
    # Excluded generic "written by" to avoid confusion with Song writers.
    "Writers": [
        r"^writer$",
        r"^writers$",
        r"^screenplay$",
        r"^screenplay by$",
        r"^story by$",
        r"^written for the screen by$",
        r"^created by$", # TV specific
        r"^based on characters created by$",
        r"^based on the novel by$",
        r"^sceneggiatura$",
        r"^sceneggiatore$",
        r"^soggetto$",
        r"^soggetto e sceneggiatura$",
        r"^guión$",
        r"^guion$",
        r"^scénario$",
        r"^d'après l'œuvre de$",
    ],

    # ==================== PRODUCERS ====================
    # Excluded "Line Producer" (moved to Prod Managers) and "VFX Producer" (moved to VFX).
    "Producers": [
        r"^producer$",
        r"^producers$",
        r"^produced by$",
        r"^executive producer$",
        r"^executive producers$",
        r"^co-?producer$",
        r"^associate producer$",
        r"^produttore$",
        r"^produttore esecutivo$",
        r"^prodotto da$",
        r"^producteur$",
        r"^producteur exécutif$",
        r"^productor$",
        r"^productor ejecutivo$",
    ],

    # ==================== CINEMATOGRAPHERS ====================
    "Cinematographers": [
        r"^cinematographer$",
        r"^cinematography$",
        r"^cinematography by$",
        r"^director of photography$",
        r"^d\.?o\.?p\.?$",
        r"^direttore della fotografia$",
        r"^fotografia$",
        r"^autori della fotografia$",
        r"^chef opérateur$",
        r"^directeur de la photographie$",
        r"^director de fotografía$",
    ],

    # ==================== EDITORS ====================
    # Strictly Picture Editors.
    "Editors": [
        r"^editor$",
        r"^film editor$",
        r"^picture editor$",
        r"^edited by$",
        r"^montaggio$",
        r"^montatore$",
        r"^chef monteur$",
        r"^montage$",
        r"^edición$",
        r"^montaje$",
    ],

    # ==================== COMPOSERS ====================
    # Strictly Score. Removed generic "Music" to avoid ambiguity with Supervisors.
    "Composers": [
        r"^composer$",
        r"^music composed by$",
        r"^original music by$",
        r"^score by$",
        r"^original score by$",
        r"^musiche$",
        r"^musiche originali$",
        r"^colonna sonora$",
        r"^musique originale$",
        r"^compositeur$",
        r"^música original$",
    ],

    # ==================== PRODUCTION DESIGNERS ====================
    "Production Designers": [
        r"^production designer$",
        r"^production design$",
        r"^production designed by$",
        r"^scenografia$",
        r"^scenografo$",
        r"^chef décorateur$",
        r"^diseño de producción$",
    ],

    # ==================== ART DIRECTORS ====================
    "Art Directors": [
        r"^art director$",
        r"^supervising art director$",
        r"^art direction$",
        r"^direzione artistica$",
        r"^directeur artistique$",
        r"^director de arte$",
    ],

    # ==================== SET DECORATORS ====================
    "Set Decorators": [
        r"^set decorator$",
        r"^set decoration$",
        r"^arredo di scena$",
        r"^arredatore$",
        r"^décorateur$",
        r"^decorador de set$",
    ],

    # ==================== COSTUME DESIGNERS ====================
    "Costume Designers": [
        r"^costume designer$",
        r"^costume design$",
        r"^designed by$", # Only if context implies costumes, otherwise risky.
        r"^costumes$",
        r"^costumes designed by$",
        r"^costumi$",
        r"^costumista$",
        r"^costumes$",
        r"^diseño de vestuario$",
        r"^vestuario$",
    ],

    # ==================== MAKEUP DEPARTMENT ====================
    "Makeup Department": [
        r"^makeup$",
        r"^make-?up$",
        r"^makeup artist$",
        r"^key makeup artist$",
        r"^makeup designer$",
        r"^hair$",
        r"^hair stylist$",
        r"^key hair stylist$",
        r"^hair designer$",
        r"^trucco$",
        r"^truccatore$",
        r"^parrucco$",
        r"^parrucchiere$",
        r"^maquillage$",
        r"^coiffure$",
        r"^maquillaje$",
        r"^peluquería$",
    ],

    # ==================== SOUND DEPARTMENT ====================
    "Sound Department": [
        r"^sound$",
        r"^sound designer$",
        r"^sound design$",
        r"^sound mixer$",
        r"^production sound mixer$",
        r"^re-recording mixer$",
        r"^supervising sound editor$",
        r"^sound editor$",
        r"^sound effects editor$",
        r"^dialogue editor$",
        r"^suono$",
        r"^fonico$",
        r"^fonico di presa diretta$",
        r"^montaggio del suono$",
        r"^ingénieur du son$",
        r"^sonido$",
        r"^diseño de sonido$",
    ],

    # ==================== VISUAL EFFECTS ====================
    "Visual Effects": [
        r"^visual effects$",
        r"^visual effects supervisor$",
        r"^visual effects producer$",
        r"^vfx supervisor$",
        r"^vfx producer$",
        r"^digital effects supervisor$",
        r"^effetti visivi$",
        r"^supervisore effetti visivi$",
        r"^effets visuels$",
        r"^efectos visuales$",
    ],

    # ==================== SPECIAL EFFECTS ====================
    "Special Effects": [
        r"^special effects$",
        r"^special effects supervisor$",
        r"^special effects coordinator$",
        r"^sfx supervisor$",
        r"^effetti speciali$",
        r"^effets spéciaux$",
        r"^efectos especiales$",
    ],

    # ==================== MUSIC DEPARTMENT ====================
    # Explicitly distinct from Composers and Soundtrack
    "Music Department": [
        r"^music supervisor$",
        r"^music editor$",
        r"^music coordinator$",
        r"^orchestration$",
        r"^orchestrator$",
        r"^conductor$",
        r"^music preparation$",
        r"^supervisione musicale$",
        r"^edizioni musicali$",
    ],

    # ==================== SOUNDTRACK (SONGS) ====================
    # Captured explicitly to avoid mixing with Score
    "Soundtrack": [
        r"^performed by$",
        r"^lyrics by$",
        r"^songs by$",
        r"^music and lyrics by$",
        r"^music produced by$", # Usually song specific
        r"^cantata da$",
        r"^interprete$",
        r"^testi di$",
    ],

    # ==================== PRODUCTION MANAGERS ====================
    "Production Managers": [
        r"^production manager$",
        r"^unit production manager$",
        r"^u\.?p\.?m\.?$",
        r"^production supervisor$",
        r"^post-production supervisor$", # Often categorized here
        r"^executive in charge of production$",
        r"^line producer$", # Moved here based on strict IMDb roles, can be swapped to Producers if preferred
        r"^direttore di produzione$",
        r"^ispettore di produzione$",
        r"^organizzatore generale$",
        r"^directeur de production$",
        r"^jefe de producción$",
    ],

    # ==================== LOCATION MANAGERS ====================
    "Location Managers": [
        r"^location manager$",
        r"^location scout$",
        r"^supervising location manager$",
        r"^location coordinator$",
        r"^location manager$",
    ],

    # ==================== CASTING DIRECTORS ====================
    "Casting Directors": [
        r"^casting by$",
        r"^casting director$",
        r"^casting$", # Risk of ambiguity, but usually implies Head
        r"^original casting$",
        r"^director de casting$",
    ],

    # ==================== SECOND UNIT / ASSISTANT DIRECTORS ====================
    "Second Unit Directors or Assistant Directors": [
        r"^assistant director$",
        r"^first assistant director$",
        r"^1st assistant director$",
        r"^1st ad$",
        r"^second assistant director$",
        r"^2nd assistant director$",
        r"^2nd ad$",
        r"^2nd unit director$",
        r"^second unit director$",
        r"^aiuto regista$",
        r"^regia seconda unità$",
        r"^assistant réalisateur$",
        r"^ayudante de dirección$",
    ],

    # ==================== CAMERA AND ELECTRICAL ====================
    "Camera and Electrical Department": [
        r"^camera operator$",
        r"^steadicam operator$",
        r"^focus puller$",
        r"^1st assistant camera$",
        r"^2nd assistant camera$",
        r"^clapper loader$",
        r"^gaffer$",
        r"^key grip$",
        r"^grip$",
        r"^best boy$",
        r"^electrician$",
        r"^lighting director$",
        r"^still photographer$",
        r"^operatore di camera$",
        r"^assistente operatore$",
        r"^capo elettricista$",
        r"^elettricista$",
        r"^macchinista$",
        r"^fotografo di scena$",
    ],

    # ==================== ART DEPARTMENT ====================
    "Art Department": [
        r"^property master$",
        r"^prop master$",
        r"^props$",
        r"^set dresser$",
        r"^leadman$",
        r"^swing gang$",
        r"^scenic artist$",
        r"^buyer$",
        r"^set decoration buyer$",
        r"^attrezzista$",
        r"^arredamento$",
        r"^accessorista$",
    ],

    # ==================== ANIMATION DEPARTMENT ====================
    "Animation Department": [
        r"^animator$",
        r"^lead animator$",
        r"^3d animator$",
        r"^2d animator$",
        r"^animation director$",
        r"^animation supervisor$",
        r"^layout artist$",
        r"^background artist$",
        r"^character design$",
        r"^title designer$", # Often goes here or VFX
        r"^title design$",
        r"^main titles$",
    ],

    # ==================== COSTUME AND WARDROBE ====================
    "Costume and Wardrobe Department": [
        r"^wardrobe$",
        r"^wardrobe supervisor$",
        r"^costume supervisor$",
        r"^key costumer$",
        r"^dresser$",
        r"^costumer$",
        r"^assistant costume designer$",
        r"^guardaroba$",
        r"^sarta di scena$",
        r"^assistente costumista$",
    ],

    # ==================== EDITORIAL DEPARTMENT ====================
    "Editorial Department": [
        r"^assistant editor$",
        r"^first assistant editor$",
        r"^additional editor$",
        r"^colorist$",
        r"^colourist$",
        r"^dailies$",
        r"^negative cutter$",
        r"^post production coordinator$",
        r"^digital intermediate$",
        r"^assistente al montaggio$",
        r"^aiuto montatore$",
        r"^colorista$",
    ],

    # ==================== SCRIPT AND CONTINUITY ====================
    "Script and Continuity Department": [
        r"^script supervisor$",
        r"^continuity$",
        r"^script coordinator$",
        r"^segretaria di edizione$",
        r"^segr. ed.$",
        r"^edizione$",
        r"^script$",
    ],

    # ==================== TRANSPORTATION ====================
    "Transportation Department": [
        r"^transportation coordinator$",
        r"^transportation captain$",
        r"^transportation manager$",
        r"^driver$",
        r"^drivers$",
        r"^transportation$",
        r"^autista$",
        r"^autisti$",
        r"^trasporti$",
    ],

    # ==================== STUNTS ====================
    "Stunts": [
        r"^stunt coordinator$",
        r"^fight coordinator$",
        r"^stunt double$",
        r"^stunts$",
        r"^stunt performer$",
        r"^maestro d'armi$",
        r"^cascatore$",
        r"^controfigura$",
    ],

    # ==================== THANKS ====================
    "Thanks": [
        r"^special thanks$",
        r"^special thanks to$",
        r"^very special thanks$",
        r"^thanks to$",
        r"^acknowledgements$",
        r"^ringraziamenti$",
        r"^ringraziamenti speciali$",
        r"^un ringraziamento a$",
    ],

    # ==================== CAST ====================
    # Very restricted to avoid false positives with crew "featuring" lists
    "Cast": [
        r"^cast$",
        r"^starring$",
        r"^main cast$",
        r"^interpreti$",
        r"^personaggi$",
        r"^con la partecipazione di$", # Be careful, usually implies cast but sometimes companies
        r"^han intervenido$",
    ],

    # ==================== DUBBING ====================
    "Dubbing": [
        r"^dubbing director$",
        r"^dubbing$",
        r"^doppiaggio$",
        r"^doppiatore$",
        r"^direzione del doppiaggio$",
        r"^versione italiana$",
        r"^edizione italiana$",
    ],

    # ==================== ADDITIONAL CREW ====================
    "Additional Crew": [
    ],
}


def normalize_role_detail(role_detail: Optional[str]) -> str:
    """Normalize role_detail for matching."""
    if not role_detail:
        return ""
    return role_detail.lower().strip()


def correct_role_group_from_detail(
    role_detail: Optional[str],
    current_role_group: str,
    enabled: bool = False
) -> Optional[str]:
    """
    Apply hard mapping correction based on EXACT role_detail match.
    
    Args:
        role_detail: The role detail text from LLM
        current_role_group: The role_group assigned by LLM
        enabled: Whether correction is enabled (default False - disabled)
        
    Returns:
        Corrected role_group if EXACT pattern matches, None otherwise
    """
    if not enabled or not role_detail:
        return None
    
    normalized_detail = normalize_role_detail(role_detail)
    if not normalized_detail:
        return None
    
    # Check each role group's patterns for EXACT match
    for target_role_group, patterns in ROLE_DETAIL_TO_ROLE_GROUP_MAPPING.items():
        for pattern in patterns:
            if re.match(pattern, normalized_detail, re.IGNORECASE):
                # Found an exact match - apply correction if different from current
                if target_role_group.lower() != current_role_group.lower():
                    logging.info(
                        f"Role correction: '{role_detail}' → "
                        f"'{current_role_group}' -> '{target_role_group}' (exact match)"
                    )
                    return target_role_group
                else:
                    # Already correct
                    return None
    
    # No exact match found - no correction
    return None


def apply_role_corrections_to_credits(
    credits: List[Dict],
    enabled: bool = False
) -> tuple[List[Dict], int]:
    """
    Apply role_group corrections to a list of credits from LLM output.
    Adds 'role_group_corrected' field when correction is applied.
    
    Args:
        credits: List of credit dictionaries from LLM
        enabled: Whether correction is enabled (default False - disabled)
        
    Returns:
        Tuple of (corrected_credits, correction_count)
    """
    if not enabled:
        return credits, 0
    
    correction_count = 0
    corrected_credits = []
    
    for credit in credits:
        role_detail = credit.get("role_detail")
        current_role_group = credit.get("role_group", "Unknown")
        
        corrected_role_group = correct_role_group_from_detail(
            role_detail,
            current_role_group,
            enabled=True
        )
        
        # Create a copy to avoid modifying original
        corrected_credit = credit.copy()
        
        if corrected_role_group:
            corrected_credit["role_group_corrected"] = corrected_role_group
            correction_count += 1
            logging.debug(
                f"Corrected: {credit.get('name')} - "
                f"{current_role_group} → {corrected_role_group}"
            )
        
        corrected_credits.append(corrected_credit)
    
    if correction_count > 0:
        logging.info(f"Applied {correction_count} role_group corrections")
    
    return corrected_credits, correction_count
