// Canonical hotel metadata + site URL constants.
//
// Single source of truth for the slug -> display-name pairing that was previously
// duplicated across app.routes.server.ts (prerender slugs) and the hotels-list
// ItemList schema. Reuse HOTEL_SLUGS / HOTELS from here instead of re-listing them.
// (public/sitemap.xml is static XML and must still be kept in sync manually.)

export const SITE_ORIGIN = 'https://buenaventuradatos.com';

// Published thesis this project comes from. Canonical URLs taken from the RIDUM
// record's own citation_* meta tags. Never use a URL carrying an
// `authentication-token` query param, that is a personal session credential and
// the PDF is public without it.
export const RESEARCH = {
  handleUrl: 'https://ridum.umanizales.edu.co/handle/20.500.12746/8174',
  pdfUrl:
    'https://ridum.umanizales.edu.co/bitstreams/f6fa7b19-82c9-470a-a845-44e9d00b8047/download',
  // Reading copy served from our own origin: RIDUM sends X-Frame-Options: DENY,
  // so its PDF cannot be embedded here. Marked noindex in public/_headers; the
  // record of version stays the RIDUM one, which is what the JSON-LD points at.
  localPdfPath: '/tesis-analisis-sentimientos-buenaventura-2026.pdf',
  repoUrl: 'https://github.com/Pipe199x/Buenaventura-Hotels',
  authorGithubUrl: 'https://github.com/Pipe199x',
  licenseUrl: 'https://creativecommons.org/licenses/by-nc-nd/4.0/deed.es',
  title: 'Análisis de sentimientos en reseñas hoteleras de Buenaventura mediante minería de texto',
  year: '2026',
} as const;

export type HotelMeta = {
  slug: string;
  displayName: string;
};

export const HOTELS: HotelMeta[] = [
  { slug: 'torre_mar', displayName: 'Hotel Torre Mar' },
  { slug: 'cosmos_pacifico', displayName: 'Hotel Cosmos Pacífico' },
  { slug: 'maguipi', displayName: 'Hotel Magüipí' },
  { slug: 'steven_buenaventura', displayName: 'Hotel Steven Buenaventura' },
  { slug: 'cordillera', displayName: 'Hotel Cordillera' },
];

export const HOTEL_SLUGS = HOTELS.map((h) => h.slug);

export function getHotelMeta(slug: string): HotelMeta | undefined {
  return HOTELS.find((h) => h.slug === slug);
}
