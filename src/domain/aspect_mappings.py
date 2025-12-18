"""
Aspect normalization mappings - Single source of truth for thematic aspect categorization.

This module defines the canonical mapping from raw aspect values (in Spanish or other languages)
to normalized thematic categories for analytical reporting.

Thematic categories use English snake_case and represent meaningful analytical groupings.
NO generic categories (other, unclassified, misc) are allowed - all aspects must map to meaningful themes.
"""
import pandas as pd
from typing import Dict, List, Set

# ==================== THEMATIC CATEGORIES ====================
# These are the high-level analytical themes for reporting
# Each theme represents a meaningful business dimension

THEMATIC_CATEGORIES = {
    "service_quality",
    "staff_attention",
    "rooms_accommodation",
    "bathrooms_cleanliness",
    "food_dining",
    "maintenance_facilities",
    "infrastructure_amenities",
    "cleanliness_general",
    "pricing_value",
    "connectivity_technology",
    "guest_experience",
    "location_surroundings",
    "safety_security",
    "noise_quietness",
    "comfort_furnishings",
}

# ==================== THEMATIC MAPPINGS ====================
# Maps raw aspect values to thematic categories
# Each theme can have multiple variants (singular/plural, synonyms, etc.)

THEMATIC_MAPPINGS: Dict[str, str] = {
    # ========== SERVICE QUALITY ==========
    "servicio": "service_quality",
    "servicios": "service_quality",
    "calidad de servicio": "service_quality",
    "atencion": "service_quality",
    "atención": "service_quality",
    "calidad": "service_quality",
    
    # ========== STAFF ATTENTION ==========
    "personal": "staff_attention",
    "empleados": "staff_attention",
    "empleado": "staff_attention",
    "recepción": "staff_attention",
    "recepcion": "staff_attention",
    "recepcionista": "staff_attention",
    "recepcionistas": "staff_attention",
    "trato": "staff_attention",
    "trato del personal": "staff_attention",
    "atención del personal": "staff_attention",
    "atencion del personal": "staff_attention",
    "amabilidad": "staff_attention",
    "cortesía": "staff_attention",
    "cortesia": "staff_attention",
    "disposición": "staff_attention",
    "disposicion": "staff_attention",
    
    # ========== ROOMS ACCOMMODATION ==========
    "habitación": "rooms_accommodation",
    "habitaciones": "rooms_accommodation",
    "habitacion": "rooms_accommodation",
    "cuarto": "rooms_accommodation",
    "cuartos": "rooms_accommodation",
    "room": "rooms_accommodation",
    "rooms": "rooms_accommodation",
    "espacio": "rooms_accommodation",
    "tamaño": "rooms_accommodation",
    "tamaño de la habitación": "rooms_accommodation",
    "tamaño habitación": "rooms_accommodation",
    "dimensiones": "rooms_accommodation",
    "vista": "rooms_accommodation",
    "vistas": "rooms_accommodation",
    "balcón": "rooms_accommodation",
    "balcon": "rooms_accommodation",
    "terraza": "rooms_accommodation",
    
    # ========== BATHROOMS CLEANLINESS ==========
    "baño": "bathrooms_cleanliness",
    "baños": "bathrooms_cleanliness",
    "bano": "bathrooms_cleanliness",
    "banos": "bathrooms_cleanliness",
    "aseo": "bathrooms_cleanliness",
    "aseos": "bathrooms_cleanliness",
    "atención aseo": "bathrooms_cleanliness",
    "atencion aseo": "bathrooms_cleanliness",
    "sanitarios": "bathrooms_cleanliness",
    "sanitario": "bathrooms_cleanliness",
    "ducha": "bathrooms_cleanliness",
    "duchas": "bathrooms_cleanliness",
    "wc": "bathrooms_cleanliness",
    "lavabo": "bathrooms_cleanliness",
    "lavabos": "bathrooms_cleanliness",
    "grifería": "bathrooms_cleanliness",
    "griferia": "bathrooms_cleanliness",
    "agua caliente": "bathrooms_cleanliness",
    "agua": "bathrooms_cleanliness",
    "presión del agua": "bathrooms_cleanliness",
    "presion del agua": "bathrooms_cleanliness",
    
    # ========== FOOD DINING ==========
    "comida": "food_dining",
    "alimentación": "food_dining",
    "alimentacion": "food_dining",
    "restaurante": "food_dining",
    "restaurantes": "food_dining",
    "desayuno": "food_dining",
    "desayunos": "food_dining",
    "almuerzo": "food_dining",
    "almuerzos": "food_dining",
    "cena": "food_dining",
    "cenas": "food_dining",
    "bebidas": "food_dining",
    "bebida": "food_dining",
    "menú": "food_dining",
    "menu": "food_dining",
    "carta": "food_dining",
    "variedad": "food_dining",
    "variedad de comida": "food_dining",
    
    # ========== MAINTENANCE FACILITIES ==========
    "mantenimiento": "maintenance_facilities",
    "reparación": "maintenance_facilities",
    "reparacion": "maintenance_facilities",
    "averías": "maintenance_facilities",
    "averias": "maintenance_facilities",
    "funcionamiento": "maintenance_facilities",
    "estado": "maintenance_facilities",
    "condición": "maintenance_facilities",
    "condicion": "maintenance_facilities",
    "conservación": "maintenance_facilities",
    "conservacion": "maintenance_facilities",
    
    # ========== INFRASTRUCTURE AMENITIES ==========
    "instalaciones": "infrastructure_amenities",
    "facilidades": "infrastructure_amenities",
    "equipamiento": "infrastructure_amenities",
    "equipo": "infrastructure_amenities",
    "infraestructura": "infrastructure_amenities",
    "piscina": "infrastructure_amenities",
    "piscinas": "infrastructure_amenities",
    "gimnasio": "infrastructure_amenities",
    "gym": "infrastructure_amenities",
    "gimnasio": "infrastructure_amenities",
    "spa": "infrastructure_amenities",
    "salón": "infrastructure_amenities",
    "salon": "infrastructure_amenities",
    "ascensor": "infrastructure_amenities",
    "ascensores": "infrastructure_amenities",
    "elevador": "infrastructure_amenities",
    "elevadores": "infrastructure_amenities",
    
    # ========== CLEANLINESS GENERAL ==========
    "limpieza": "cleanliness_general",
    "limpieza general": "cleanliness_general",
    "higiene": "cleanliness_general",
    "higiene general": "cleanliness_general",
    "orden": "cleanliness_general",
    "aseo general": "cleanliness_general",
    "condiciones sanitarias": "cleanliness_general",
    
    # ========== PRICING VALUE ==========
    "precio": "pricing_value",
    "precios": "pricing_value",
    "costo": "pricing_value",
    "costos": "pricing_value",
    "tarifa": "pricing_value",
    "tarifas": "pricing_value",
    "valor": "pricing_value",
    "relación calidad precio": "pricing_value",
    "relacion calidad precio": "pricing_value",
    "calidad precio": "pricing_value",
    "precio calidad": "pricing_value",
    "relación precio": "pricing_value",
    "relacion precio": "pricing_value",
    "económico": "pricing_value",
    "economico": "pricing_value",
    "caro": "pricing_value",
    "cara": "pricing_value",
    "barato": "pricing_value",
    "barata": "pricing_value",
    
    # ========== CONNECTIVITY TECHNOLOGY ==========
    "wifi": "connectivity_technology",
    "wi-fi": "connectivity_technology",
    "internet": "connectivity_technology",
    "conexión": "connectivity_technology",
    "conexion": "connectivity_technology",
    "red": "connectivity_technology",
    "señal": "connectivity_technology",
    "senal": "connectivity_technology",
    "cobertura": "connectivity_technology",
    "televisión": "connectivity_technology",
    "television": "connectivity_technology",
    "tv": "connectivity_technology",
    "televisor": "connectivity_technology",
    
    # ========== GUEST EXPERIENCE ==========
    "experiencia": "guest_experience",
    "estancia": "guest_experience",
    "estadía": "guest_experience",
    "estadia": "guest_experience",
    "visita": "guest_experience",
    "recomendación": "guest_experience",
    "recomendacion": "guest_experience",
    "satisfacción": "guest_experience",
    "satisfaccion": "guest_experience",
    "impresión": "guest_experience",
    "impresion": "guest_experience",
    "ambiente": "guest_experience",
    "atmósfera": "guest_experience",
    "atmosfera": "guest_experience",
    
    # ========== LOCATION SURROUNDINGS ==========
    "ubicación": "location_surroundings",
    "ubicacion": "location_surroundings",
    "localización": "location_surroundings",
    "localizacion": "location_surroundings",
    "zona": "location_surroundings",
    "barrio": "location_surroundings",
    "cercanía": "location_surroundings",
    "cercania": "location_surroundings",
    "acceso": "location_surroundings",
    "accesos": "location_surroundings",
    "entorno": "location_surroundings",
    "alrededores": "location_surroundings",
    "cerca": "location_surroundings",
    "lejos": "location_surroundings",
    "playa": "location_surroundings",
    "centro": "location_surroundings",
    
    # ========== SAFETY SECURITY ==========
    "seguridad": "safety_security",
    "safe": "safety_security",
    "seguro": "safety_security",
    "protección": "safety_security",
    "proteccion": "safety_security",
    "vigilancia": "safety_security",
    "cámaras": "safety_security",
    "camaras": "safety_security",
    "cerradura": "safety_security",
    "cerraduras": "safety_security",
    "caja fuerte": "safety_security",
    "caja de seguridad": "safety_security",
    
    # ========== NOISE QUIETNESS ==========
    "ruido": "noise_quietness",
    "ruidos": "noise_quietness",
    "sonido": "noise_quietness",
    "silencioso": "noise_quietness",
    "tranquilo": "noise_quietness",
    "tranquilidad": "noise_quietness",
    "silencio": "noise_quietness",
    "molestias": "noise_quietness",
    "molestia": "noise_quietness",
    
    # ========== COMFORT FURNISHINGS ==========
    "cama": "comfort_furnishings",
    "camas": "comfort_furnishings",
    "colchón": "comfort_furnishings",
    "colchones": "comfort_furnishings",
    "colchon": "comfort_furnishings",
    "colchones": "comfort_furnishings",
    "almohada": "comfort_furnishings",
    "almohadas": "comfort_furnishings",
    "comodidad": "comfort_furnishings",
    "confort": "comfort_furnishings",
    "mobiliario": "comfort_furnishings",
    "muebles": "comfort_furnishings",
    "mueble": "comfort_furnishings",
    "sábanas": "comfort_furnishings",
    "sabanas": "comfort_furnishings",
    "toallas": "comfort_furnishings",
    "toalla": "comfort_furnishings",
    
    # ========== AIR CONDITIONING (mapped to infrastructure) ==========
    "aire": "infrastructure_amenities",
    "aire acondicionado": "infrastructure_amenities",
    "a/c": "infrastructure_amenities",
    "ac": "infrastructure_amenities",
    "climatización": "infrastructure_amenities",
    "climatizacion": "infrastructure_amenities",
    "clima": "infrastructure_amenities",
    "ventilación": "infrastructure_amenities",
    "ventilacion": "infrastructure_amenities",
    "ventilador": "infrastructure_amenities",
    "ventiladores": "infrastructure_amenities",
    "calefacción": "infrastructure_amenities",
    "calefaccion": "infrastructure_amenities",
    
    # ========== PARKING (mapped to infrastructure) ==========
    "estacionamiento": "infrastructure_amenities",
    "parqueadero": "infrastructure_amenities",
    "parqueo": "infrastructure_amenities",
    "parking": "infrastructure_amenities",
    "aparcamiento": "infrastructure_amenities",
    "estacionar": "infrastructure_amenities",
}

# ==================== SEMANTIC FALLBACK RULES ====================
# Keywords that help assign unmapped aspects to meaningful themes
# Used when direct mapping fails

SEMANTIC_KEYWORDS: Dict[str, str] = {
    # Service-related keywords
    "servicio": "service_quality",
    "atencion": "service_quality",
    "atención": "service_quality",
    "trato": "staff_attention",
    "personal": "staff_attention",
    "empleado": "staff_attention",
    
    # Room-related keywords
    "habitacion": "rooms_accommodation",
    "habitación": "rooms_accommodation",
    "cuarto": "rooms_accommodation",
    "room": "rooms_accommodation",
    
    # Cleanliness-related keywords
    "limpieza": "cleanliness_general",
    "aseo": "bathrooms_cleanliness",
    "bano": "bathrooms_cleanliness",
    "baño": "bathrooms_cleanliness",
    "higiene": "cleanliness_general",
    
    # Food-related keywords
    "comida": "food_dining",
    "restaurante": "food_dining",
    "desayuno": "food_dining",
    
    # Infrastructure keywords
    "instalacion": "infrastructure_amenities",
    "instalación": "infrastructure_amenities",
    "equipo": "infrastructure_amenities",
    "aire": "infrastructure_amenities",
    
    # Price-related keywords
    "precio": "pricing_value",
    "costo": "pricing_value",
    "tarifa": "pricing_value",
    
    # Location keywords
    "ubicacion": "location_surroundings",
    "ubicación": "location_surroundings",
    "zona": "location_surroundings",
    
    # Experience keywords
    "experiencia": "guest_experience",
    "estancia": "guest_experience",
}


def _normalize_text(text: str) -> str:
    """
    Normalize text for matching: lowercase, strip, remove accents.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Convert to lowercase and strip
    text = str(text).lower().strip()
    
    # Remove accents (simple mapping for common Spanish characters)
    accent_map = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n",
        "Á": "a", "É": "e", "Í": "i", "Ó": "o", "Ú": "u", "Ñ": "n",
    }
    for accented, unaccented in accent_map.items():
        text = text.replace(accented, unaccented)
    
    return text


def get_thematic_aspect(raw_aspect: str) -> str:
    """
    Get thematic category for a raw aspect value.
    
    Uses a multi-step matching strategy:
    1. Direct mapping (exact match after normalization)
    2. Partial matching (substring match)
    3. Semantic keyword matching
    4. Fallback to high-level meaningful theme (guest_experience)
    
    Args:
        raw_aspect: Raw aspect string (may contain Spanish, accents, etc.)
        
    Returns:
        Thematic category name in English (snake_case)
        NEVER returns generic categories like "other" or "unclassified"
    """
    if not raw_aspect or pd.isna(raw_aspect):
        # Fallback to meaningful theme for empty values
        return "guest_experience"
    
    # Normalize input
    normalized = _normalize_text(raw_aspect)
    
    # Step 1: Direct match
    if normalized in THEMATIC_MAPPINGS:
        return THEMATIC_MAPPINGS[normalized]
    
    # Step 2: Partial match (check if any mapping key is contained in the input)
    for variant, theme in THEMATIC_MAPPINGS.items():
        if variant in normalized or normalized in variant:
            return theme
    
    # Step 3: Semantic keyword matching
    for keyword, theme in SEMANTIC_KEYWORDS.items():
        if keyword in normalized:
            return theme
    
    # Step 4: Fallback to meaningful high-level theme
    # Analyze the text to determine the most appropriate high-level category
    if any(word in normalized for word in ["servicio", "atencion", "trato", "personal"]):
        return "service_quality"
    elif any(word in normalized for word in ["habitacion", "cuarto", "room", "espacio"]):
        return "rooms_accommodation"
    elif any(word in normalized for word in ["bano", "baño", "aseo", "ducha"]):
        return "bathrooms_cleanliness"
    elif any(word in normalized for word in ["comida", "restaurante", "desayuno"]):
        return "food_dining"
    elif any(word in normalized for word in ["precio", "costo", "tarifa", "valor"]):
        return "pricing_value"
    elif any(word in normalized for word in ["ubicacion", "ubicación", "zona", "lugar"]):
        return "location_surroundings"
    elif any(word in normalized for word in ["limpieza", "higiene", "orden"]):
        return "cleanliness_general"
    elif any(word in normalized for word in ["instalacion", "equipo", "infraestructura"]):
        return "infrastructure_amenities"
    elif any(word in normalized for word in ["experiencia", "estancia", "visita"]):
        return "guest_experience"
    else:
        # Final fallback: assign to guest_experience (meaningful, not generic)
        return "guest_experience"


def add_thematic_mapping(variant: str, theme: str) -> None:
    """
    Add a new aspect mapping dynamically.
    
    Args:
        variant: Raw aspect variant
        theme: Thematic category (must be in THEMATIC_CATEGORIES)
    """
    if theme not in THEMATIC_CATEGORIES:
        raise ValueError(
            f"Thematic category '{theme}' not in allowed set: {THEMATIC_CATEGORIES}. "
            "Generic categories (other, unclassified, misc) are not allowed."
        )
    
    normalized_variant = _normalize_text(variant)
    THEMATIC_MAPPINGS[normalized_variant] = theme


def get_all_variants(theme: str) -> List[str]:
    """
    Get all known variants for a thematic category.
    
    Args:
        theme: Thematic category name
        
    Returns:
        List of variant strings
    """
    return [variant for variant, t in THEMATIC_MAPPINGS.items() if t == theme]


def get_thematic_statistics() -> Dict[str, int]:
    """
    Get count of variants per thematic category.
    
    Returns:
        Dictionary mapping theme to variant count
    """
    stats = {}
    for theme in THEMATIC_CATEGORIES:
        stats[theme] = len(get_all_variants(theme))
    return stats
