# Web Deployment Plan (Cloudflare Pages + Supabase)

This playbook turns the Buenaventura hotel insights dashboard into a public site using your Cloudflare-owned domain, Supabase URL, anon key, and existing SQL views. The goal is to stay in free tiers while keeping a path to future growth.

## Target Architecture
- **Frontend:** Static site built with **Vite + React** (or Next.js static export if you prefer). React keeps the bundle light and plays nicely with Supabase JS and charting libs.
- **Hosting:** **Cloudflare Pages** for global CDN, automatic HTTPS, and Git-based auto-deploys on every push to `main`.
- **Data layer:** **Supabase Postgres** with your existing SQL views; browser reads are powered by the **anon** key.
- **Optional server logic:** **Supabase Edge Functions** (only if you later need writes, scheduled jobs, or service-role access). Keep the service key out of the browser.
- **Assets:** Small static assets can live in the repo; larger files/images go in a **public Supabase Storage bucket** with cached URLs.

## Quick Start Checklist
- [ ] Confirm you have: Cloudflare domain access, Supabase URL, Supabase anon key, and names of the SQL views you want to expose.
- [ ] Create or reuse a `frontend/` folder with the Vite + React app.
- [ ] Add environment variables locally and in Cloudflare Pages.
- [ ] Deploy via Cloudflare Pages and wire the domain CNAME.
- [ ] Smoke-test charts against Supabase views.

## Environment Variables
Set these in **Cloudflare Pages → Project → Settings → Environment Variables**:
- `VITE_SUPABASE_URL` – your Supabase project URL.
- `VITE_SUPABASE_ANON_KEY` – public anon key (safe for browser use).

For local development, place them in `frontend/.env` (never commit secrets):
```
VITE_SUPABASE_URL=...your-url...
VITE_SUPABASE_ANON_KEY=...your-anon-key...
```

## Frontend Scaffold (Vite + React)
1) Create the app (inside `frontend/`):
   ```bash
   npm create vite@latest frontend -- --template react
   cd frontend
   npm install
   ```
2) Add Supabase client at `src/supabaseClient.ts`:
   ```ts
   import { createClient } from '@supabase/supabase-js'

   const supabaseUrl = import.meta.env.VITE_SUPABASE_URL!
   const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY!

   export const supabase = createClient(supabaseUrl, supabaseAnonKey)
   ```
3) Query a view in a component (swap the view + columns to match yours):
   ```ts
   import { useEffect, useState } from 'react'
   import { supabase } from './supabaseClient'

   type ViewRow = { hotel: string; year: number; sentiment_score: number }

   export function SentimentByYear() {
     const [rows, setRows] = useState<ViewRow[]>([])
     const [error, setError] = useState<string | null>(null)

     useEffect(() => {
       supabase
         .from('vw_sentiment_by_year')
         .select('hotel, year, sentiment_score')
         .order('year')
         .then(({ data, error }) => {
           if (error) setError(error.message)
           else setRows(data ?? [])
         })
     }, [])

     if (error) return <div>Failed to load data: {error}</div>

     return (
       <table>
         <thead>
           <tr><th>Hotel</th><th>Year</th><th>Sentiment</th></tr>
         </thead>
         <tbody>
           {rows.map((row, idx) => (
             <tr key={idx}>
               <td>{row.hotel}</td>
               <td>{row.year}</td>
               <td>{row.sentiment_score.toFixed(2)}</td>
             </tr>
           ))}
         </tbody>
       </table>
     )
   }
   ```
4) Charts: `recharts` or `react-chartjs-2` are light and friendly. Keep datasets small by selecting only needed columns and paginating where applicable.
5) Local run: `npm run dev` (Vite) and open the shown localhost URL.

### Project Layout (example)
```
repo-root/
  frontend/
    src/
      supabaseClient.ts
      components/
        SentimentByYear.tsx
        ...
    index.html
    package.json
    vite.config.ts
  docs/
    WEB_DEPLOYMENT_PLAN.md
```

## Deployment on Cloudflare Pages
1) Push the `frontend/` folder to the repo (or keep it here already).
2) In Cloudflare Pages:
   - Create a new project and connect this GitHub repo.
   - **Build command:** `npm run build`
   - **Build output directory:** `dist`
   - **Root directory:** `frontend` (so Pages runs the build there).
   - Add the environment variables listed above.
3) Deploy. Pages gives you a `*.pages.dev` URL with HTTPS by default.
4) Enable **auto-deploys on `main`** so every push refreshes the site.

## Domain Wiring (Cloudflare)
1) In Cloudflare DNS, add a **CNAME**: `preview--buenaventura-hotel-insights.lovable.app` → `{your-project}.pages.dev`.
2) In Cloudflare Pages → Custom domains, add `preview--buenaventura-hotel-insights.lovable.app` and complete validation.
3) Enable **Always Use HTTPS** and **Automatic HTTPS Rewrites** in Cloudflare SSL/TLS.
4) Verify after DNS propagation (usually minutes on Cloudflare-managed zones).

## Data Security and RLS
- Use only the **anon key** in the browser. Keep the **service role key** exclusively in Supabase Edge Functions (if/when you add them).
- For read-only dashboards, enable RLS on underlying tables and views:
  ```sql
  alter table your_table enable row level security;
  create policy "Public read-only for anon" on your_table
    for select to anon using (true);
  ```
- If a view joins multiple tables, ensure each source table has equivalent `select` policies.
- Consider a **throttling policy** in the UI (e.g., debounce search inputs) to avoid excessive free-tier usage.

## Performance and Cost Control
- Vite’s hashed filenames provide automatic cache busting; Cloudflare Pages will cache globally.
- Query only the columns needed per chart/view and prefer server-side ordering/aggregation in your SQL views.
- Add database indexes on frequent filter columns in your views to keep latency low.
- Use **Supabase Storage** for larger images; serve via public bucket URLs and let Cloudflare cache them.

## Observability and Troubleshooting
- **Supabase logs:** Use the dashboard to watch query errors/latency. Add `explain analyze` on slow views and add indexes as needed.
- **Cloudflare Pages logs:** Check build logs for dependency errors and deployment status. Redeploy from the Pages UI if needed.
- **Browser monitoring:** Keep an eye on the console for Supabase errors; surface user-friendly error messages (as in the sample component above).

## Rollout Steps (minimal path to live)
1) Add/confirm `.env` locally and wire Supabase URL/anon key.
2) Build + preview locally: `npm run dev` and hit the views you plan to show.
3) Push to GitHub; let Cloudflare Pages build and deploy from `frontend/`.
4) Point the CNAME to the Pages domain and verify HTTPS.
5) Smoke-test the public URL: check table renders, chart renders, and no console errors.
