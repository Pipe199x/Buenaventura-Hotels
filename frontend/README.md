# Buenaventura Hotel Insights – Frontend

A lightweight Vite + React dashboard that reads Supabase SQL views with the public anon key. It is sized for Cloudflare Pages free tier and points to your Supabase project via environment variables.

## Prerequisites
- Node.js 18+
- Supabase project URL and **anon** key (browser-safe). Do **not** use the service role in the browser.

## Local development
1. Copy `.env.example` to `.env` and fill with your Supabase values:
   ```bash
   cp .env.example .env
   ```
2. Install dependencies and start Vite:
   ```bash
   npm install
   npm run dev
   ```
3. Open the printed localhost URL and confirm the table/chart render data from your view.

## Deployment (Cloudflare Pages)
- **Root directory:** `frontend`
- **Build command:** `npm run build`
- **Build output:** `dist`
- **Environment variables:**
  - `VITE_SUPABASE_URL` (e.g., `https://...supabase.co`)
  - `VITE_SUPABASE_ANON_KEY` (anon key)
  - Optional: `VITE_SENTIMENT_VIEW` if you want a different Supabase view name.
- Connect your Git repo, enable auto-deploys on `main`, and map the Cloudflare domain (e.g., `preview--buenaventura-hotel-insights.lovable.app`) via CNAME.

## Customization
- Swap the query in `src/components/SentimentByYear.tsx` for any other view/columns.
- Adjust styles in `src/styles.css`.
- Add more charts using `recharts` or your preferred charting library.
