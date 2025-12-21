# Web Deployment Plan (Cloudflare Pages + Supabase)

This plan explains how to publish the Buenaventura hotel insights dashboard on the public internet using the existing Supabase database and SQL views. It favors free-tier services and keeps costs near zero.

## Target Architecture
- **Frontend:** Static site generated with **Vite + React** (free to host). Any modern Vite template works; React has first-class Supabase JS support and a rich charting ecosystem.
- **Hosting:** **Cloudflare Pages** for global CDN, HTTPS, and free SSL. Git-based CI triggers redeploys on every push to `main`.
- **Backend:** **Supabase Postgres** with existing SQL views. All read-only queries use the public `anon` key from the browser.
- **Optional server logic:** Supabase **Edge Functions** only if you later need privileged operations (keep outside the browser and use the service role key only there).
- **Analytics assets:** Store static images or downloadable reports either inside the repo (if small) or in a **public Supabase Storage bucket**.

## Environment Variables
Configure the following in Cloudflare Pages → Project → Settings → Environment Variables:
- `VITE_SUPABASE_URL` – your Supabase project URL.
- `VITE_SUPABASE_ANON_KEY` – public anon key (safe for browser use).

For local development, create a `.env` file in the frontend project root:
```
VITE_SUPABASE_URL=...your-url...
VITE_SUPABASE_ANON_KEY=...your-anon-key...
```

## Frontend Scaffold (Vite + React)
1. On your machine (or Codespaces), create the app next to this repo or inside a new `frontend/` folder:
   ```bash
   npm create vite@latest frontend -- --template react
   cd frontend
   npm install
   ```
2. Add Supabase client setup in `src/supabaseClient.ts`:
   ```ts
   import { createClient } from '@supabase/supabase-js'

   const supabaseUrl = import.meta.env.VITE_SUPABASE_URL!
   const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY!

   export const supabase = createClient(supabaseUrl, supabaseAnonKey)
   ```
3. Query your SQL views from a React component (example):
   ```ts
   import { useEffect, useState } from 'react'
   import { supabase } from './supabaseClient'

   type ViewRow = { hotel: string; year: number; sentiment_score: number }

   export function SentimentByYear() {
     const [rows, setRows] = useState<ViewRow[]>([])

     useEffect(() => {
       supabase.from('vw_sentiment_by_year').select('*').then(({ data, error }) => {
         if (error) console.error(error)
         else setRows(data ?? [])
       })
     }, [])

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
4. Add charts with a lightweight library (e.g., `chart.js` + `react-chartjs-2` or `recharts`) and connect to other views in the same pattern.

## Deployment on Cloudflare Pages
1. Commit the frontend code to this repo (e.g., under `frontend/`).
2. In Cloudflare Pages:
   - Create a new project connected to this GitHub repo.
   - **Build command:** `npm run build` (set the working directory to `frontend/`).
   - **Build output directory:** `dist`.
   - Set the environment variables noted above.
3. Trigger a deployment (initial push or manual). Cloudflare will produce a public URL and automatically provision HTTPS.

## Domain Wiring (Cloudflare)
1. In your Cloudflare DNS, create a **CNAME** record for `preview--buenaventura-hotel-insights.lovable.app` pointing to the Cloudflare Pages domain (e.g., `your-project.pages.dev`).
2. Enable **Always Use HTTPS** and **Automatic HTTPS Rewrites** in Cloudflare SSL/TLS settings.
3. Wait for DNS propagation, then verify the custom domain in Cloudflare Pages → Custom domains.

## Data Security and Access
- Keep the **service role key** out of the browser; only use the **anon key** for public, read-only views.
- Enable **Row Level Security (RLS)** on tables. For read-only public dashboards, create policies that allow `select` for the `anon` role only on safe views/tables.
- If you need authenticated dashboards later, integrate Supabase Auth and gate components on `session` state.

## Observability and Cost Control
- Use Supabase project logs to monitor query errors and latency on the views the dashboard calls.
- Cloudflare Pages provides deployment logs and caching via the CDN; cache busting is handled automatically by Vite hash filenames.
- Keep images/assets small to stay within free transfer limits.

## Next Steps Checklist
- [ ] Scaffold frontend with Vite + React and add Supabase client.
- [ ] Wire components to your SQL views for tables and charts.
- [ ] Configure Cloudflare Pages project and environment variables.
- [ ] Point the custom domain CNAME to the Pages domain and verify HTTPS.
- [ ] Smoke test dashboard end-to-end (data loads, charts render, no console errors).
