import { RenderMode, ServerRoute } from '@angular/ssr';

const HOTEL_SLUGS = [
  'torre_mar',
  'cosmos_pacifico',
  'maguipi',
  'steven_buenaventura',
  'cordillera',
] as const;

export const serverRoutes: ServerRoute[] = [
  // Rutas estáticas
  { path: '', renderMode: RenderMode.Prerender },
  { path: 'home', renderMode: RenderMode.Prerender },
  { path: 'about', renderMode: RenderMode.Prerender },
  { path: 'hotels', renderMode: RenderMode.Prerender },

  // ✅ Ruta dinámica prerender (necesita params)
  {
    path: 'hotels/:hotelSlug',
    renderMode: RenderMode.Prerender,
    getPrerenderParams: async () =>
      HOTEL_SLUGS.map((hotelSlug) => ({ hotelSlug })),
  },

  // Fallback
  { path: '**', renderMode: RenderMode.Prerender },
];
