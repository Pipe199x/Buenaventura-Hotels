-- Schema backup of the Supabase objects behind buenaventuradatos.com (project rxdsgkkwfbbfnthusunr).
-- Generated 2026-10-11 from the live catalog: tables, constraints, RLS, policies, vw_* views, grants.
-- No data and no secrets. The data backup is kept out of git: Datasets/supabase_backup_2026-10-11/*.parquet
--
-- Restore into a fresh project:
--   1. Run this file in the SQL editor (as postgres: the views must be owned by postgres so they can
--      read the RLS-protected tables, which is how the site reads aggregates with the anon key).
--   2. Load each parquet file into the table of the same name (columns match 1:1).
--   3. select setval('public.hotel_negative_topics_yearly_id_seq', (select max(id) from public.hotel_negative_topics_yearly));

begin;

create sequence public.hotel_negative_topics_yearly_id_seq as bigint start with 1 increment by 1 minvalue 1 maxvalue 9223372036854775807 cache 1;

create table public.hotels_gold (
  hotel_id text,
  reviewid text not null,
  placeid text,
  title text,
  text text,
  texttranslated text,
  originallanguage text,
  revieworigin text,
  publishedatdate timestamp with time zone,
  year_month text,
  stars double precision,
  totalscore double precision,
  reviewscount double precision,
  hotelstars double precision,
  price double precision,
  islocalguide boolean,
  reviewernumberofreviews double precision,
  likescount double precision,
  responsefromownertext text,
  responsefromownerdate timestamp with time zone,
  response_delay_days double precision,
  review_length integer,
  scrapedat timestamp with time zone,
  categoryname text,
  reviewurl text,
  url text,
  "reviewContext/Food & drinks" text,
  "reviewContext/Hotel highlights" text,
  "reviewContext/Nearby activities" text,
  "reviewContext/Noteworthy details" text,
  "reviewContext/Rooms" text,
  "reviewContext/Safety" text,
  "reviewContext/Travel group" text,
  "reviewContext/Trip type" text,
  "reviewContext/Walkability" text,
  "reviewDetailedRating/Location" double precision,
  "reviewDetailedRating/Rooms" double precision,
  "reviewDetailedRating/Service" double precision,
  text_used text,
  sentiment_label text,
  positive_score double precision,
  neutral_score double precision,
  negative_score double precision,
  sentiment_score double precision,
  sentences_count integer,
  aspects text,
  key_phrases text,
  entities text,
  pii_entities text,
  linked_entities text,
  scored_at timestamp with time zone,
  hotel_name text,
  constraint hotels_gold_pkey PRIMARY KEY (reviewid)
);

create table public.hotel_negative_topics_yearly (
  id bigint default nextval('hotel_negative_topics_yearly_id_seq'::regclass) not null,
  hotel_name text not null,
  thematic_label text not null,
  year integer not null,
  frequency integer not null,
  constraint hotel_negative_topics_yearly_pkey PRIMARY KEY (id),
  constraint uq_hotel_negative_topic_year UNIQUE (hotel_name, thematic_label, year)
);

alter sequence public.hotel_negative_topics_yearly_id_seq owned by public.hotel_negative_topics_yearly.id;

alter table public.hotels_gold enable row level security;
create policy "read hotels_gold for authenticated" on public.hotels_gold as permissive for select to authenticated using (true);
alter table public.hotel_negative_topics_yearly enable row level security;

create view public.vw_home_hotels_cards_count as
 WITH overview AS (
         SELECT hotels_gold.hotel_name,
            count(*) AS total_reviews_hotel,
            round(avg(hotels_gold.stars)::numeric, 2)::double precision AS avg_star_rating_hotel
           FROM hotels_gold
          WHERE hotels_gold.publishedatdate >= '2020-01-01 00:00:00+00'::timestamp with time zone AND hotels_gold.publishedatdate < '2025-08-22 00:00:00+00'::timestamp with time zone AND (hotels_gold.revieworigin IS NULL OR hotels_gold.revieworigin = 'Google'::text)
          GROUP BY hotels_gold.hotel_name
        ), polarity AS (
         SELECT hotels_gold.hotel_name,
            hotels_gold.sentiment_label
           FROM hotels_gold
          WHERE hotels_gold.publishedatdate IS NOT NULL AND hotels_gold.publishedatdate >= '2020-01-01 00:00:00+00'::timestamp with time zone AND hotels_gold.publishedatdate < '2025-08-22 00:00:00+00'::timestamp with time zone AND (hotels_gold.revieworigin IS NULL OR hotels_gold.revieworigin = 'Google'::text)
          GROUP BY hotels_gold.hotel_name, (date_trunc('month'::text, hotels_gold.publishedatdate)), hotels_gold.sentiment_label
        ), satisfaction AS (
         SELECT polarity.hotel_name,
            round(100.0 * count(*) FILTER (WHERE polarity.sentiment_label = 'positive'::text)::numeric / NULLIF(count(*) FILTER (WHERE polarity.sentiment_label IS NOT NULL AND btrim(polarity.sentiment_label) <> ''::text AND (polarity.sentiment_label = ANY (ARRAY['positive'::text, 'neutral'::text, 'negative'::text]))), 0)::numeric, 2)::double precision AS satisfaction_rate_hotel
           FROM polarity
          GROUP BY polarity.hotel_name
        ), exploded AS (
         SELECT hg.hotel_name,
            lower(TRIM(BOTH FROM regexp_replace(split_part(p.p, '('::text, 1), ':\s*$'::text, ''::text, 'g'::text))) AS raw_aspect
           FROM hotels_gold hg,
            LATERAL unnest(string_to_array(hg.aspects, '|'::text)) p(p)
          WHERE hg.aspects IS NOT NULL AND btrim(hg.aspects) <> ''::text AND hg.publishedatdate >= '2020-01-01 00:00:00+00'::timestamp with time zone AND hg.publishedatdate < '2025-08-22 00:00:00+00'::timestamp with time zone AND (hg.revieworigin IS NULL OR hg.revieworigin = 'Google'::text)
        ), normalized AS (
         SELECT exploded.hotel_name,
                CASE
                    WHEN exploded.raw_aspect = ANY (ARRAY['abitacion'::text, 'abitaciones'::text, 'habitación'::text, 'habitaciones'::text, 'habitacion'::text]) THEN 'habitacion'::text
                    ELSE exploded.raw_aspect
                END AS aspect_clean
           FROM exploded
          WHERE exploded.raw_aspect IS NOT NULL AND exploded.raw_aspect <> ''::text AND exploded.raw_aspect <> 'hotel'::text
        ), counts AS (
         SELECT normalized.hotel_name,
            normalized.aspect_clean,
            count(*) AS total_mentions
           FROM normalized
          GROUP BY normalized.hotel_name, normalized.aspect_clean
        ), ranked AS (
         SELECT counts.hotel_name,
            counts.aspect_clean,
            counts.total_mentions,
            row_number() OVER (PARTITION BY counts.hotel_name ORDER BY counts.total_mentions DESC) AS rn
           FROM counts
        ), top5 AS (
         SELECT ranked.hotel_name,
            string_agg(ranked.aspect_clean, ', '::text ORDER BY ranked.total_mentions DESC) AS top_words
           FROM ranked
          WHERE ranked.rn <= 5
          GROUP BY ranked.hotel_name
        )
 SELECT o.hotel_name,
        CASE o.hotel_name
            WHEN 'torre_mar'::text THEN 'Hotel Torre Mar'::text
            WHEN 'cosmos_pacifico'::text THEN 'Hotel Cosmos Pacífico'::text
            WHEN 'maguipi'::text THEN 'Hotel Magüipi'::text
            WHEN 'steven_buenaventura'::text THEN 'Hotel Steven Buenaventura'::text
            WHEN 'cordillera'::text THEN 'Hotel Cordillera'::text
            ELSE o.hotel_name
        END AS hotel_display_name,
    o.avg_star_rating_hotel,
    o.total_reviews_hotel,
    COALESCE(s.satisfaction_rate_hotel, 0::double precision) AS satisfaction_rate_hotel,
        CASE o.hotel_name
            WHEN 'torre_mar'::text THEN 'Cl. 1 #3 03, Buenaventura, Valle del Cauca'::text
            WHEN 'cosmos_pacifico'::text THEN 'Cl. 3 #1A - 57, Buenaventura, Valle del Cauca'::text
            WHEN 'maguipi'::text THEN 'Sector Rural, Vereda Punta San Pedro, Buenaventura, Valle del Cauca'::text
            WHEN 'steven_buenaventura'::text THEN 'Cra. 4 #442, Buenaventura, Valle del Cauca'::text
            WHEN 'cordillera'::text THEN 'Dg. 3a #3b - 51, Centro, Buenaventura, Valle del Cauca'::text
            ELSE NULL::text
        END AS address,
    COALESCE(t.top_words, ''::text) AS top_words
   FROM overview o
     LEFT JOIN satisfaction s ON s.hotel_name = o.hotel_name
     LEFT JOIN top5 t ON t.hotel_name = o.hotel_name
  ORDER BY o.total_reviews_hotel DESC;

create view public.vw_home_overview_global_count as
 WITH per_hotel AS (
         SELECT hotels_gold.hotel_name,
            count(*) AS total_reviews,
            round(avg(hotels_gold.stars)::numeric, 2)::double precision AS avg_star_rating
           FROM hotels_gold
          WHERE hotels_gold.publishedatdate >= '2020-01-01 00:00:00+00'::timestamp with time zone AND hotels_gold.publishedatdate < '2025-08-22 00:00:00+00'::timestamp with time zone AND (hotels_gold.revieworigin IS NULL OR hotels_gold.revieworigin = 'Google'::text)
          GROUP BY hotels_gold.hotel_name
        ), overview AS (
         SELECT sum(per_hotel.total_reviews)::bigint AS total_reviews,
            round(avg(per_hotel.avg_star_rating)::numeric, 2)::double precision AS avg_star_rating
           FROM per_hotel
        ), polarity AS (
         SELECT hotels_gold.hotel_name,
            hotels_gold.sentiment_label
           FROM hotels_gold
          WHERE hotels_gold.publishedatdate IS NOT NULL AND hotels_gold.publishedatdate >= '2020-01-01 00:00:00+00'::timestamp with time zone AND hotels_gold.publishedatdate < '2025-08-22 00:00:00+00'::timestamp with time zone AND (hotels_gold.revieworigin IS NULL OR hotels_gold.revieworigin = 'Google'::text)
          GROUP BY hotels_gold.hotel_name, (date_trunc('month'::text, hotels_gold.publishedatdate)), hotels_gold.sentiment_label
        ), satisfaction AS (
         SELECT round(100.0 * count(*) FILTER (WHERE polarity.sentiment_label = 'positive'::text)::numeric / NULLIF(count(*) FILTER (WHERE polarity.sentiment_label IS NOT NULL AND btrim(polarity.sentiment_label) <> ''::text), 0)::numeric, 2)::double precision AS tasa_satisfaccion_pct
           FROM polarity
        )
 SELECT o.total_reviews,
    o.avg_star_rating,
    s.tasa_satisfaccion_pct
   FROM overview o
     CROSS JOIN satisfaction s;

create view public.vw_home_sentiment_distribution_count as
 WITH polarity AS (
         SELECT hotels_gold.hotel_name,
            hotels_gold.sentiment_label
           FROM hotels_gold
          WHERE hotels_gold.publishedatdate IS NOT NULL AND hotels_gold.publishedatdate >= '2020-01-01 00:00:00+00'::timestamp with time zone AND hotels_gold.publishedatdate < '2025-08-22 00:00:00+00'::timestamp with time zone AND (hotels_gold.revieworigin IS NULL OR hotels_gold.revieworigin = 'Google'::text)
          GROUP BY hotels_gold.hotel_name, (date_trunc('month'::text, hotels_gold.publishedatdate)), hotels_gold.sentiment_label
        ), base AS (
         SELECT polarity.hotel_name,
            polarity.sentiment_label
           FROM polarity
          WHERE polarity.sentiment_label IS NOT NULL AND btrim(polarity.sentiment_label) <> ''::text AND (polarity.sentiment_label = ANY (ARRAY['positive'::text, 'neutral'::text, 'negative'::text]))
        ), counts AS (
         SELECT base.hotel_name,
            base.sentiment_label,
            count(*) AS label_count
           FROM base
          GROUP BY base.hotel_name, base.sentiment_label
        ), totals AS (
         SELECT counts.hotel_name,
            sum(counts.label_count)::bigint AS total_labeled
           FROM counts
          GROUP BY counts.hotel_name
        )
 SELECT c.hotel_name,
        CASE c.hotel_name
            WHEN 'torre_mar'::text THEN 'Hotel Torre Mar'::text
            WHEN 'cosmos_pacifico'::text THEN 'Hotel Cosmos Pacífico'::text
            WHEN 'maguipi'::text THEN 'Hotel Magüipi'::text
            WHEN 'steven_buenaventura'::text THEN 'Hotel Steven Buenaventura'::text
            WHEN 'cordillera'::text THEN 'Hotel Cordillera'::text
            ELSE c.hotel_name
        END AS hotel_display_name,
    c.sentiment_label,
    round(100.0 * (c.label_count::numeric / NULLIF(t.total_labeled, 0)::numeric), 2)::double precision AS pct_label
   FROM counts c
     JOIN totals t ON t.hotel_name = c.hotel_name
  ORDER BY c.hotel_name, (
        CASE c.sentiment_label
            WHEN 'positive'::text THEN 1
            WHEN 'neutral'::text THEN 2
            WHEN 'negative'::text THEN 3
            ELSE 9
        END);

create view public.vw_hotel_response_rate as
 SELECT hotel_name,
    count(*) AS total_reviews,
    count(*) FILTER (WHERE responsefromownertext IS NOT NULL AND btrim(responsefromownertext) <> ''::text) AS total_reviews_responded,
    round(count(*) FILTER (WHERE responsefromownertext IS NOT NULL AND btrim(responsefromownertext) <> ''::text)::numeric / NULLIF(count(*), 0)::numeric * 100::numeric, 2) AS pct_reviews_responded
   FROM hotels_gold
  GROUP BY hotel_name
  ORDER BY hotel_name;

create view public.vw_negative_topics_table as
 SELECT hotel_name,
    thematic_label,
    COALESCE(sum(frequency) FILTER (WHERE year = 2020), 0::bigint) AS y2020,
    COALESCE(sum(frequency) FILTER (WHERE year = 2021), 0::bigint) AS y2021,
    COALESCE(sum(frequency) FILTER (WHERE year = 2022), 0::bigint) AS y2022,
    COALESCE(sum(frequency) FILTER (WHERE year = 2023), 0::bigint) AS y2023,
    COALESCE(sum(frequency) FILTER (WHERE year = 2024), 0::bigint) AS y2024,
    COALESCE(sum(frequency) FILTER (WHERE year = 2025), 0::bigint) AS y2025,
    sum(frequency) AS total,
    count(*) FILTER (WHERE frequency > 0) AS active_years
   FROM hotel_negative_topics_yearly
  GROUP BY hotel_name, thematic_label
  ORDER BY hotel_name, (count(*) FILTER (WHERE frequency > 0)) DESC, (sum(frequency)) DESC;

create view public.vw_trend_line as
 WITH base AS (
         SELECT hotels_gold.hotel_name,
            EXTRACT(year FROM hotels_gold.publishedatdate)::integer AS year,
            hotels_gold.sentiment_label,
            count(*) AS total_reviews
           FROM hotels_gold
          WHERE hotels_gold.sentiment_label IS NOT NULL AND hotels_gold.publishedatdate >= '2020-01-01 00:00:00+00'::timestamp with time zone AND hotels_gold.publishedatdate < '2026-01-01 00:00:00+00'::timestamp with time zone
          GROUP BY hotels_gold.hotel_name, (EXTRACT(year FROM hotels_gold.publishedatdate)::integer), hotels_gold.sentiment_label
        )
 SELECT hotel_name,
    year,
    sentiment_label,
    total_reviews,
    sum(total_reviews) OVER (PARTITION BY hotel_name, year) AS total_year_reviews,
    round(total_reviews::numeric / sum(total_reviews) OVER (PARTITION BY hotel_name, year), 4) AS pct_year_decimal
   FROM base
  ORDER BY hotel_name, year, sentiment_label;

grant all on table public.vw_home_hotels_cards_count to anon;
grant all on table public.vw_home_hotels_cards_count to authenticated;
-- grant select on table public.vw_home_hotels_cards_count to report_reader;  -- custom role: create it first
grant all on table public.vw_home_hotels_cards_count to service_role;
grant all on table public.vw_home_overview_global_count to anon;
grant all on table public.vw_home_overview_global_count to authenticated;
-- grant select on table public.vw_home_overview_global_count to report_reader;  -- custom role: create it first
grant all on table public.vw_home_overview_global_count to service_role;
grant all on table public.vw_home_sentiment_distribution_count to anon;
grant all on table public.vw_home_sentiment_distribution_count to authenticated;
-- grant select on table public.vw_home_sentiment_distribution_count to report_reader;  -- custom role: create it first
grant all on table public.vw_home_sentiment_distribution_count to service_role;
grant all on table public.vw_hotel_response_rate to anon;
grant all on table public.vw_hotel_response_rate to authenticated;
-- grant select on table public.vw_hotel_response_rate to report_reader;  -- custom role: create it first
grant all on table public.vw_hotel_response_rate to service_role;
grant all on table public.vw_negative_topics_table to anon;
grant all on table public.vw_negative_topics_table to authenticated;
-- grant select on table public.vw_negative_topics_table to report_reader;  -- custom role: create it first
grant all on table public.vw_negative_topics_table to service_role;
grant all on table public.vw_trend_line to anon;
grant all on table public.vw_trend_line to authenticated;
-- grant select on table public.vw_trend_line to report_reader;  -- custom role: create it first
grant all on table public.vw_trend_line to service_role;
grant all on table public.hotel_negative_topics_yearly to anon;
grant all on table public.hotel_negative_topics_yearly to authenticated;
-- grant select on table public.hotel_negative_topics_yearly to report_reader;  -- custom role: create it first
grant all on table public.hotel_negative_topics_yearly to service_role;
grant all on table public.hotels_gold to anon;
grant all on table public.hotels_gold to authenticated;
-- grant select on table public.hotels_gold to report_reader;  -- custom role: create it first
grant all on table public.hotels_gold to service_role;
grant all on sequence public.hotel_negative_topics_yearly_id_seq to anon;
grant all on sequence public.hotel_negative_topics_yearly_id_seq to authenticated;
grant all on sequence public.hotel_negative_topics_yearly_id_seq to service_role;

commit;
