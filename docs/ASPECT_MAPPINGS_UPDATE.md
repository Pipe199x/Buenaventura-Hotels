# Aspect Mappings Update - All Hotels

## Summary

Updated `src/domain/aspect_mappings.py` to include comprehensive mappings for all hotels based on manual thematic analysis tables. This ensures consistent categorization across all hotels (Cosmos Pacífico, Cordillera, Steven Buenaventura, Torre Mar, and Magüipí).

## New Mappings Added

### Food & Dining
- `alimentos`, `calidad de alimentos`, `manejo de vajilla`, `vajilla`, `platos`, `mesa`, `bar`, `tienda`
- Covers: "restaurante / alimentos", "calidad de alimentos", "manejo de vajilla", "restaurante - mesa"

### Staff Attention
- `personal del hotel`, `trato atención del personal`, `consideración con huéspedes`, `anfitriones`
- Covers: "trato / atención del personal", "personal del hotel", "consideración con huéspedes", "personal / anfitriones"

### Service Quality
- `atención servicio`, `servicio atención`, `organización`, `coordinación`, `publicidad engañosa`, `logística`
- Covers: "servicio / atención", "organización", "publicidad engañosa", "logística / organización"

### Rooms & Accommodation
- `habitación confort`, `vista entorno`, `paisaje`, `cabañas`
- Covers: "habitación / confort", "vista / entorno", "cabañas / habitaciones"

### Infrastructure & Amenities
- `instalaciones generales`, `infraestructura general`, `infraestructura mantenimiento`, `zonas húmedas`, `áreas comunes`, `ambiente`, `atmósfera`, `tranquilidad`, `actividades`, `planes`
- Covers: "instalaciones / infraestructura", "infraestructura general", "áreas comunes", "piscinas / zonas húmedas", "actividades / planes", "ambiente / tranquilidad"

### Maintenance & Facilities
- `mant`, `pintura`, `paredes`, `plomería`, `baño plomería`, `plagas`, `baño plagas`
- Covers: "pintura", "paredes", "baño / plomería", "baño / plagas"

### Connectivity & Technology
- `internet conectividad`, `teléfono`, `citófono`, `dispositivos`, `tarjeta`, `equipos tecnología`, `tecnología`, `comunicación interna`, `facturación electrónica`
- Covers: "teléfono / teléfonos", "citófonos", "dispositivos (teléfono/tarjeta)", "equipos / tecnología", "comunicación interna", "facturación electrónica", "wifi / internet"

### Comfort & Furnishings
- `camas colchones`, `lencería`, `sábanas lencería`
- Covers: "camas / colchones", "sábanas / lencería"

### Location & Surroundings
- `ubicación entorno`, `entorno natural`, `playas`, `mar`, `charcos`, `lugar`, `sitio`, `lugar sitio`, `muelle`, `muelle turístico`, `puerto`, `paisaje`, `cultura`, `cultura playas entorno`
- Covers: "ubicación / entorno", "lugar / sitio", "playas / mar / charcos", "paisajes / vistas / entorno natural", "cultura / playas / entorno"

### Guest Experience
- `experiencia general`, `experiencia general estadía`, `experiencia negativa`, `experiencia global`, `impresión general`, `percepción general`, `percepción general del hotel`, `hotel`, `perfil huésped`, `familias`, `vacaciones`, `accesibilidad`, `servicios adicionales`
- Covers: "experiencia general / estadía", "experiencia negativa", "percepción general del hotel", "hotel", "perfil huésped (familias/vacaciones)", "accesibilidad", "servicios adicionales (spa/eco)"

### Cleanliness General
- `aseo limpieza`, `olores`, `olor`, `olores olor`
- Covers: "aseo / limpieza", "olores / olor"

### Noise & Quietness
- `música`, `música ambiente`
- Covers: "música / ambiente"

## Impact

This update ensures that:
1. **All hotels** have consistent aspect categorization
2. **All manual thematic categories** from the analysis tables are properly mapped
3. **No aspects fall into generic categories** - everything maps to meaningful themes
4. **Year-by-year comparisons** will be accurate across all hotels
5. **Future reviews** will be correctly categorized using the comprehensive mapping

## Testing

After this update, re-run the Gold layer processing for all hotels to ensure:
- All aspects are correctly mapped
- Thematic tables match the manual analysis
- No aspects are left unmapped (falling back to `guest_experience`)

## Next Steps

1. Re-process all hotels through the Gold layer
2. Verify thematic tables match manual analysis
3. Update any hotel-specific documentation if needed

