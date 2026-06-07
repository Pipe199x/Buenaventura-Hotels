import { RenderMode, ServerRoute } from '@angular/ssr';
import { HOTEL_SLUGS } from './core/seo/hotels.metadata';

export const serverRoutes: ServerRoute[] = [
  // Pre-render static pages.
  { path: '', renderMode: RenderMode.Prerender },
  { path: 'home', renderMode: RenderMode.Prerender },
  { path: 'about', renderMode: RenderMode.Prerender },
  { path: 'hotels', renderMode: RenderMode.Prerender },

  // Pre-render hotel detail pages for known slugs.
  {
    path: 'hotels/:hotelSlug',
    renderMode: RenderMode.Prerender,
    getPrerenderParams: async () =>
      HOTEL_SLUGS.map((hotelSlug) => ({ hotelSlug })),
  },

  // Fallback for any unmatched path.
  { path: '**', renderMode: RenderMode.Prerender },
];
