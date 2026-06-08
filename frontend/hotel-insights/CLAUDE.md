# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
npm start                       # Dev server (ng serve) at http://localhost:4200
npm run build                   # Production build = prerender every route to static HTML (see Deploy below)
npm run watch                   # Rebuild on change (development config)
npm test                        # Run unit tests (Vitest via @angular/build:unit-test)
npm run serve:ssr:hotel-insights # Run the built SSR Express server (after a build)
npx ng test --include='**/home.spec.ts'  # Run a single spec
```

There is no separate lint step configured. Formatting is Prettier (config lives in `package.json`: 100 cols, single quotes, `angular` parser for `.html`).

## Big-picture architecture

This is an **Angular 21 standalone (no NgModules) SSR app** — a read-only Spanish-language
analytics dashboard for hotel review sentiment in Buenaventura. Data comes from **Supabase**
(Postgres); the app is **prerendered and deployed to Netlify**.

### Data layer — Supabase views only
All data reads go through Postgres **views prefixed `vw_`** (e.g. `vw_home_overview_global_count`,
`vw_home_hotels_cards_count`, `vw_trend_line`). The canonical access point is
`src/app/core/data/home-data.service.ts`, which owns the row types and every query. The single
Supabase client is created in `src/app/core/supabase/supabase.client.ts` from
`src/environments/environment.ts` (Supabase URL + anon key live there; there is **no prod
environment file replacement**, so `environment.ts` is used in all builds). The app never writes
data — it only `select`s from views.

### Zone gotcha (critical)
Supabase promises resolve **outside Angular's zone**, so change detection does not fire
automatically after an `await`. Components that load data manually re-enter the zone and/or force
detection: see `hotel-detail.ts` (`NgZone.run(...)` + `ChangeDetectorRef.detectChanges()`) and
`home.ts` / `hotels-list.ts` (`detectChanges()` around the load). **Follow this pattern when
adding any component that awaits Supabase**, or the UI will silently fail to update.

### Auth — overlay modal, not routes
There are **no `/login` or `/register` routes**. Authentication is a single global overlay dialog:
- `AuthService` (`src/app/core/auth.service.ts`) wraps Supabase auth and exposes `session$`
  (a `BehaviorSubject`); it is initialized once via `APP_INITIALIZER` in `app.config.ts`.
- `AuthModalService` + `AuthModalComponent` (`src/app/shared/auth-modal/`) render a tabbed
  login/register dialog, mounted once in `app.component.ts`. Open it with
  `authModal.open('login' | 'register', redirectUrl?)` from anywhere (navbar, hotel cards).
- OAuth preserves the post-login destination across the page reload via `sessionStorage`
  (`authRedirect`), handled in `AuthModalComponent.ngOnInit`.
- Supabase auth errors are mapped to Spanish in `src/app/core/supabase/auth-errors.ts`
  (`translateAuthError`) and surfaced via `ToastService` (`src/app/shared/toast/`), whose host
  lives in `app.component.ts`. There are **no route guards** — gated UI is a soft prompt only.

### Routing & shell
`app.routes.ts` renders all pages inside `ShellComponent` (`src/app/shared/shell/`), which provides
the `Navbar` + layout. The navbar is session-aware (login button vs. logout) via `session$`.

### SSR / prerender — hardcoded hotel slugs
`src/app/app.routes.server.ts` prerenders the static pages and the hotel detail pages for a
**hardcoded `HOTEL_SLUGS` array**. When adding/removing a hotel you must update that list, and the
same slugs/names are duplicated in the JSON-LD `ItemList` in `hotels-list.ts`, in
`public/sitemap.xml`, and in `public/llms.txt` — keep all four in sync.

### Charts
Charts use **ngx-echarts**. ECharts is tree-shaken: required pieces are registered with
`echarts.use([...])` in `app.config.ts`. **To use a new chart type/component you must add it there**
or it will render blank. Chart components live in `src/app/shared/charts/`.

### SEO
`SchemaService` (`src/app/core/seo/schema.service.ts`) injects/removes `application/ld+json` script
tags in the document head by id. Route components set their structured data on init (see
`hotels-list.ts`).

### Deploy — prerendered (SSG) static hosting
`npm run build` runs the production build (`outputMode: server`), which **prerenders every
route** in `app.routes.server.ts` to static HTML under `dist/hotel-insights/browser/`
(`index.html` = home, `about/index.html`, `hotels/<slug>/index.html`, …). Netlify deploys
that `browser/` folder as a **static site** — no Node runtime needed. `public/_redirects`
provides the SPA fallback (`/* /index.html 200`); Netlify serves the prerendered file for
known routes and only rewrites genuinely unknown paths. Client hydration is enabled
(`provideClientHydration` in `app.config.ts`), so the prerendered DOM is reused, not
re-rendered. `src/server.ts` + the `server/` output remain available for optional on-demand
SSR (`serve:ssr:hotel-insights`) but are not used by the static deploy.

> Do **not** clobber the prerendered `index.html` with `index.csr.html` — that empties the
> home page for crawlers (the old `netlify-postbuild.mjs` did this; it was removed).

## Conventions & gotchas

- **Standalone components everywhere**; state is RxJS (`BehaviorSubject` / observables), not signals.
- **File naming**: feature components are suffix-less (`home.ts` / `home.html` / `home.scss`, class
  `Home`); services are `*.service.ts`. Newer shared UI uses the `Component` suffix
  (`ToastComponent`, `AuthModalComponent`).
- **Dead scaffolding — do not edit by mistake**: `src/app/app.ts` (class `App`) with `app.html`/
  `app.scss`, and `src/app/features/hotels/hotels.ts` (class `Hotels`) are leftovers from
  `ng generate` and are **not wired into routing**. The real root — bootstrapped by **both**
  `main.ts` (browser) and `main.server.ts` (SSR/prerender) — is `src/app/app.component.ts`
  (`AppComponent`). Both entry points must bootstrap the same root or hydration breaks.
- **Styling**: pure SCSS, component-scoped, no UI framework. Shared design tokens: primary blue
  `#3b82f6`, success `#2e7d32`, neutral/amber `#f9a825`, negative `#c62828`. Reuse these (and the
  toast/modal styles) rather than introducing a component library.
- **UI copy is Spanish.**
