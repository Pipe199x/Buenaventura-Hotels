# Web Application Deployment Guide

Complete guide for deploying the hotel reviews analytics web application to Cloudflare.

## Overview

The web application consists of:
- **Frontend**: Static HTML/CSS/JavaScript files
- **Backend API**: Cloudflare Pages Functions (serverless)
- **Database**: Supabase PostgreSQL

## Prerequisites

1. Cloudflare account (free tier works)
2. Supabase project with `hotels_gold` table
3. GitHub/GitLab repository (for Cloudflare Pages)

## Step 1: Prepare Your Code

1. **Ensure your database is populated:**
   ```bash
   # Run the pipeline to load data to Supabase
   python -m src.nashor_to_supabase --hotel cordillera
   # Repeat for all hotels
   ```

2. **Verify database structure:**
   - Table: `hotels_gold` (or `parsed`)
   - Required columns: `hotel_name`, `sentiment_label`, `sentiment_score`, `stars`, `aspect_theme`, `publishedAtDate`

## Step 2: Deploy to Cloudflare Pages

### Option A: Via Cloudflare Dashboard

1. **Go to Cloudflare Dashboard:**
   - Navigate to **Pages** → **Create a project**
   - Connect your Git repository

2. **Configure Build Settings:**
   - **Framework preset**: None (static site)
   - **Build command**: (leave empty)
   - **Build output directory**: `web`
   - **Root directory**: `/` (or `web` if you want to deploy only web folder)

3. **Set Environment Variables:**
   - Go to **Settings** → **Environment Variables**
   - Add:
     - `SUPABASE_URL`: `https://your-project.supabase.co`
     - `SUPABASE_API_KEY`: Your Supabase anon key (from Supabase dashboard)

4. **Deploy:**
   - Click **Save and Deploy**
   - Wait for deployment to complete

### Option B: Via Wrangler CLI

1. **Install Wrangler:**
   ```bash
   npm install -g wrangler
   ```

2. **Login:**
   ```bash
   wrangler login
   ```

3. **Deploy:**
   ```bash
   cd web
   wrangler pages deploy . --project-name=hotel-reviews
   ```

4. **Set Secrets:**
   ```bash
   wrangler pages secret put SUPABASE_URL
   wrangler pages secret put SUPABASE_API_KEY
   ```

## Step 3: Configure Custom Domain (Optional)

1. **In Cloudflare Pages:**
   - Go to **Custom domains**
   - Click **Set up a custom domain**
   - Enter your domain (e.g., `analytics.yourdomain.com`)

2. **Update DNS:**
   - Cloudflare will provide DNS records
   - Add CNAME record pointing to your Pages URL
   - SSL certificate will be automatically provisioned

## Step 4: Test the Application

1. **Visit your deployment URL:**
   - Cloudflare Pages URL: `https://your-project.pages.dev`
   - Or your custom domain

2. **Test features:**
   - Select a hotel from dropdown
   - Click "Cargar Datos"
   - Verify charts and tables load correctly

## Step 5: Update API Endpoint (If Needed)

If you deployed API separately as a Worker:

1. **Get Worker URL:**
   - From Cloudflare Workers dashboard
   - Copy the worker URL (e.g., `https://hotel-reviews-api.your-subdomain.workers.dev`)

2. **Update `web/app.js`:**
   ```javascript
   const API_BASE_URL = 'https://hotel-reviews-api.your-subdomain.workers.dev/api';
   ```

## Troubleshooting

### API Returns 404

- Check that `/functions/api/[[path]].js` exists
- Verify environment variables are set
- Check Cloudflare Pages Functions logs

### Database Connection Errors

- Verify Supabase URL and API key are correct
- Check Supabase project is active
- Ensure database table exists and has data

### CORS Errors

- CORS headers are already configured in the API
- If issues persist, check browser console for specific errors

### Charts Not Loading

- Verify Chart.js CDN is accessible
- Check browser console for JavaScript errors
- Ensure API is returning data in correct format

## Performance Optimization

1. **Enable Caching:**
   - Cloudflare automatically caches static assets
   - API responses can be cached with appropriate headers

2. **Optimize Database Queries:**
   - Add indexes on frequently queried columns
   - Consider materialized views for complex queries

3. **Reduce Bundle Size:**
   - Use Chart.js from CDN (already done)
   - Minify CSS/JS for production

## Security Considerations

1. **API Keys:**
   - Never expose Supabase service role key
   - Use anon key for public access
   - Consider Row Level Security (RLS) in Supabase

2. **Rate Limiting:**
   - Cloudflare automatically provides DDoS protection
   - Consider adding rate limits to API endpoints

3. **Data Privacy:**
   - Ensure no PII is exposed in API responses
   - Review what data is publicly accessible

## Monitoring

1. **Cloudflare Analytics:**
   - View traffic in Cloudflare Dashboard
   - Monitor API usage and errors

2. **Supabase Dashboard:**
   - Monitor database query performance
   - Check for slow queries

## Updates and Maintenance

1. **Update Code:**
   - Push changes to Git repository
   - Cloudflare Pages will automatically rebuild

2. **Update Environment Variables:**
   - Go to Pages → Settings → Environment Variables
   - Update values as needed
   - Redeploy if necessary

## Support

For issues:
1. Check Cloudflare Pages Functions logs
2. Check browser console for errors
3. Verify Supabase connection
4. Review this documentation

