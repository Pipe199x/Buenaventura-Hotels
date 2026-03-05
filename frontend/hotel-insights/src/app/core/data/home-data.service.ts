import { Injectable } from '@angular/core';
import { supabase } from '../supabase/supabase.client';

/* =======================
   TIPOS – HOME OVERVIEW
======================= */

export type HomeOverviewGlobal = {
  total_reviews: number;
  avg_star_rating: number;
  tasa_satisfaccion_pct: number;
};

/* =======================
   TIPOS – HOTEL CARDS
======================= */

export type HotelCardRow = {
  hotel_name: string;
  hotel_display_name: string;
  avg_star_rating_hotel: number;
  total_reviews_hotel: number;
  satisfaction_rate_hotel: number;
  address: string | null;
  top_words: string;
};

/* =======================
   TIPOS – CHART BARRAS
======================= */

export type SentimentDistributionRow = {
  hotel_name: string;
  hotel_display_name: string;
  sentiment_label: 'positive' | 'neutral' | 'negative';
  pct_label: number;
};

/* =======================
   SERVICE
======================= */

@Injectable({ providedIn: 'root' })
export class HomeDataService {
  /* =======================
     KPIs GLOBALES (HOME)
  ======================= */
  async getHomeOverviewGlobal(): Promise<HomeOverviewGlobal> {
    const { data, error } = await supabase
      .from('vw_home_overview_global_count')
      .select('total_reviews, avg_star_rating, tasa_satisfaccion_pct')
      .single();

    if (error) {
      console.error('Error cargando KPIs globales', error);
      throw error;
    }

    return data as HomeOverviewGlobal;
  }

  /* =======================
     CARDS DE HOTELES
  ======================= */
  async getHotelCards(): Promise<HotelCardRow[]> {
    const { data, error } = await supabase
      .from('vw_home_hotels_cards_count')
      .select(`
        hotel_name,
        hotel_display_name,
        avg_star_rating_hotel,
        total_reviews_hotel,
        satisfaction_rate_hotel,
        address,
        top_words
      `)
      .order('total_reviews_hotel', { ascending: false });

    if (error) {
      console.error('Error cargando cards de hoteles', error);
      throw error;
    }

    return (data ?? []) as HotelCardRow[];
  }

  /* =======================
     DISTRIBUCIÓN SENTIMIENTOS
     (GRÁFICO BARRAS)
  ======================= */
  async getSentimentDistribution(): Promise<SentimentDistributionRow[]> {
    const { data, error } = await supabase
      .from('vw_home_sentiment_distribution_count')
      .select(`
        hotel_name,
        hotel_display_name,
        sentiment_label,
        pct_label
      `);

    if (error) {
      console.error('Error cargando distribución de sentimientos', error);
      throw error;
    }

    return (data ?? []) as SentimentDistributionRow[];
  }
}
