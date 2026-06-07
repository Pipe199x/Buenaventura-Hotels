// Pure builders for hotel-related schema.org (JSON-LD) objects.
// Shared by the hotels list (ItemList) and the hotel detail page.

import { HotelMeta, SITE_ORIGIN } from './hotels.metadata';

// Subset of HotelCardRow with the fields used to enrich the schema.
export type HotelSchemaData = {
  address?: string | null;
  avg_star_rating_hotel?: number | null;
  total_reviews_hotel?: number | null;
};

export type BreadcrumbItem = {
  name: string;
  url: string;
};

// Builds a schema.org Hotel object. `data` is optional: when omitted (or missing
// rating/address) only the static base fields are emitted, which is what gets
// captured during prerender. The post-load pass passes `data` to add the
// PostalAddress and AggregateRating.
export function buildHotelSchema(meta: HotelMeta, data?: HotelSchemaData): Record<string, unknown> {
  const url = `${SITE_ORIGIN}/hotels/${meta.slug}`;

  const schema: Record<string, unknown> = {
    '@type': 'Hotel',
    '@id': `${url}#hotel`,
    name: meta.displayName,
    url,
    isPartOf: { '@id': `${SITE_ORIGIN}/#website` },
  };

  const street = data?.address?.trim();
  if (street) {
    schema['address'] = {
      '@type': 'PostalAddress',
      streetAddress: street,
      addressLocality: 'Buenaventura',
      addressRegion: 'Valle del Cauca',
      addressCountry: 'CO',
    };
  }

  const rating = Number(data?.avg_star_rating_hotel ?? 0);
  const reviewCount = Number(data?.total_reviews_hotel ?? 0);
  if (rating > 0 && reviewCount > 0) {
    schema['aggregateRating'] = {
      '@type': 'AggregateRating',
      ratingValue: Number(rating.toFixed(2)),
      reviewCount,
      bestRating: 5,
      worstRating: 1,
    };
  }

  return schema;
}

// Builds a schema.org BreadcrumbList from an ordered list of crumbs.
export function buildBreadcrumb(items: BreadcrumbItem[]): Record<string, unknown> {
  return {
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: items.map((item, i) => ({
      '@type': 'ListItem',
      position: i + 1,
      name: item.name,
      item: item.url,
    })),
  };
}
