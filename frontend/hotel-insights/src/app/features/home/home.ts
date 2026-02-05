import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';

import {
  HomeDataService,
  HomeOverviewGlobal,
  HotelCardRow,
  SentimentDistributionRow,
} from '../../core/data/home-data.service';

// ✅ IMPORT del componente del chart (ajusta la ruta si tu carpeta difiere)
import { SentimentStackedBarComponent } from '../../shared/charts/sentiment-stacked-bar/sentiment-stacked-bar';

type BadgeTone = 'positive' | 'neutral' | 'negative';

type HotelBadge = {
  pct: number; // 0-100
  tone: BadgeTone;
  label: 'Positiva' | 'Neutral' | 'Negativa';
};

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [CommonModule, SentimentStackedBarComponent], // ✅ IMPORTA el chart aquí
  templateUrl: './home.html',
  styleUrl: './home.scss',
})
export class Home implements OnInit {
  loading = true;
  errorMsg = '';

  kpis: HomeOverviewGlobal | null = null;
  hotels: HotelCardRow[] = [];

  // ✅ data del chart
  sentimentRows: SentimentDistributionRow[] = [];
  chartLoading = true;

  // badge por hotel_name
  hotelBadge = new Map<string, HotelBadge>();

  constructor(private homeData: HomeDataService) {}

  async ngOnInit() {
    this.loading = true;
    this.chartLoading = true;
    this.errorMsg = '';

    try {
      const [kpis, hotels, dist] = await Promise.all([
        this.homeData.getHomeOverviewGlobal(),
        this.homeData.getHotelCards(),
        this.homeData.getSentimentDistribution(),
      ]);

      this.kpis = kpis;
      this.hotels = hotels;

      this.sentimentRows = dist ?? [];
      this.hotelBadge = this.buildBadges(this.sentimentRows);
    } catch (e: any) {
      this.errorMsg = e?.message ?? 'Error cargando datos del Home';
    } finally {
      this.loading = false;
      this.chartLoading = false;
    }
  }

  // chips (con "+X más")
  asChips(topWords: string, max = 6): { items: string[]; more: number } {
    if (!topWords) return { items: [], more: 0 };
    const all = topWords
      .split(',')
      .map((w) => w.trim())
      .filter(Boolean);

    const items = all.slice(0, max);
    const more = Math.max(0, all.length - items.length);
    return { items, more };
  }

  // badge fallback si por algo no hay dist
  getBadgeForHotel(hotelName: string): HotelBadge {
    return (
      this.hotelBadge.get(hotelName) ?? {
        pct: 0,
        tone: 'neutral',
        label: 'Neutral',
      }
    );
  }

  private buildBadges(rows: SentimentDistributionRow[]): Map<string, HotelBadge> {
    // agrupa por hotel_name y toma el MAYOR pct_label
    const byHotel = new Map<string, SentimentDistributionRow[]>();

    for (const r of rows) {
      if (!byHotel.has(r.hotel_name)) byHotel.set(r.hotel_name, []);
      byHotel.get(r.hotel_name)!.push(r);
    }

    const out = new Map<string, HotelBadge>();

    for (const [hotelName, list] of byHotel.entries()) {
      const best = [...list].sort((a, b) => (b.pct_label ?? 0) - (a.pct_label ?? 0))[0];

      const tone = best?.sentiment_label ?? 'neutral';
      const pct = Math.round(best?.pct_label ?? 0);

      const label =
        tone === 'positive'
          ? 'Positiva'
          : tone === 'negative'
          ? 'Negativa'
          : 'Neutral';

      out.set(hotelName, { pct, tone, label });
    }

    return out;
  }
}
