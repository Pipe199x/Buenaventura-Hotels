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
   TIPOS – TREND LINE
======================= */

export type SentimentTrendRow = {
  hotel_name: string;
  year: number;
  sentiment_label: 'positive' | 'neutral' | 'negative';
  total_reviews: number;
  total_year_reviews: number;
  pct_year_decimal: number;
};

/* =======================
   TIPOS – RESPONSE RATE
======================= */

export type HotelResponseRateRow = {
  hotel_name: string;
  total_reviews: number;
  total_reviews_responded: number;
  pct_reviews_responded: number;
};

/* =======================
   SERVICE
======================= */

@Injectable({ providedIn: 'root' })
export class HomeDataService {
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

  async getSentimentTrendByHotel(hotelName: string): Promise<SentimentTrendRow[]> {
    const { data, error } = await supabase
      .from('vw_trend_line')
      .select(`
        hotel_name,
        year,
        sentiment_label,
        total_reviews,
        total_year_reviews,
        pct_year_decimal
      `)
      .eq('hotel_name', hotelName)
      .order('year', { ascending: true })
      .order('sentiment_label', { ascending: true });

    if (error) {
      console.error('Error cargando tendencia anual de sentimiento', error);
      throw error;
    }

    return (data ?? []) as SentimentTrendRow[];
  }

  async getHotelResponseRateByHotel(hotelName: string): Promise<HotelResponseRateRow | null> {
    const { data, error } = await supabase
      .from('vw_hotel_response_rate')
      .select(`
        hotel_name,
        total_reviews,
        total_reviews_responded,
        pct_reviews_responded
      `)
      .eq('hotel_name', hotelName)
      .maybeSingle();

    if (error) {
      console.error('Error cargando tasa de respuesta del hotel', error);
      throw error;
    }

    return data as HotelResponseRateRow | null;
  }
}
