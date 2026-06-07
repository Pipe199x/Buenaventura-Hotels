// Canonical hotel metadata + site URL constants.
//
// Single source of truth for the slug -> display-name pairing that was previously
// duplicated across app.routes.server.ts (prerender slugs) and the hotels-list
// ItemList schema. Reuse HOTEL_SLUGS / HOTELS from here instead of re-listing them.
// (public/sitemap.xml is static XML and must still be kept in sync manually.)

export const SITE_ORIGIN = 'https://buenaventuradatos.com';

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
