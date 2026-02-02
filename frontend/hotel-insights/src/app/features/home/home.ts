import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { NgxEchartsModule } from 'ngx-echarts';

import {
  HomeDataService,
  HomeOverviewGlobal,
  HotelCardRow,
  SentimentDistributionRow
} from '../../core/data/home-data.service';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [CommonModule, NgxEchartsModule],
  templateUrl: './home.html',
  styleUrl: './home.scss',
})
export class Home implements OnInit {
  loading = true;
  errorMsg = '';

  kpis: HomeOverviewGlobal | null = null;
  hotels: HotelCardRow[] = [];

  chartOption: any = null;
  chartLoading = true;

  constructor(
    private homeData: HomeDataService,
    private cdr: ChangeDetectorRef
  ) {}

  async ngOnInit() {
    this.loading = true;
    this.chartLoading = true;
    this.errorMsg = '';

    const results = await Promise.allSettled([
      this.homeData.getHomeOverviewGlobal(),
      this.homeData.getHotelCards(),
      this.homeData.getSentimentDistribution(),
    ]);

    const kpisRes = results[0];
    const hotelsRes = results[1];
    const distRes = results[2];

    if (kpisRes.status === 'fulfilled') this.kpis = kpisRes.value;
    else console.error('KPIs error', kpisRes.reason);

    if (hotelsRes.status === 'fulfilled') this.hotels = hotelsRes.value;
    else console.error('Hotels error', hotelsRes.reason);

    if (distRes.status === 'fulfilled') {
      this.buildGroupedBars(distRes.value);
    } else {
      console.error('Chart error', distRes.reason);
      this.chartOption = null;
    }

    const firstError =
      (kpisRes.status === 'rejected' && kpisRes.reason) ||
      (hotelsRes.status === 'rejected' && hotelsRes.reason) ||
      (distRes.status === 'rejected' && distRes.reason);

    if (firstError) {
      this.errorMsg = firstError?.message ?? 'Error cargando datos del Home';
    }

    this.loading = false;
    this.chartLoading = false;

    // ✅ CLAVE en zoneless: forzar render cuando termina async
    this.cdr.detectChanges();
  }

  asChips(topWords: string): string[] {
    if (!topWords) return [];
    return topWords.split(',').map(w => w.trim()).filter(Boolean);
  }

  private buildGroupedBars(rows: SentimentDistributionRow[]) {
    const hotels = Array.from(new Set(rows.map(r => r.hotel_display_name)));

    const positive = new Map<string, number>();
    const neutral  = new Map<string, number>();
    const negative = new Map<string, number>();

    for (const r of rows) {
      if (r.sentiment_label === 'positive') positive.set(r.hotel_display_name, r.pct_label);
      if (r.sentiment_label === 'neutral')  neutral.set(r.hotel_display_name, r.pct_label);
      if (r.sentiment_label === 'negative') negative.set(r.hotel_display_name, r.pct_label);
    }

    const posData = hotels.map(h => positive.get(h) ?? 0);
    const neuData = hotels.map(h => neutral.get(h) ?? 0);
    const negData = hotels.map(h => negative.get(h) ?? 0);

    this.chartOption = {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params: any[]) => {
          const hotel = params?.[0]?.axisValue ?? '';
          const lines = params.map(p => `${p.marker} ${p.seriesName}: ${p.value}%`);
          return `<b>${hotel}</b><br/>${lines.join('<br/>')}`;
        }
      },
      legend: { top: 0, data: ['Positivo', 'Neutral', 'Negativo'] },
      grid: { left: 40, right: 20, top: 40, bottom: 80, containLabel: true },
      xAxis: {
        type: 'category',
        data: hotels,
        axisLabel: { interval: 0, rotate: 25, hideOverlap: false }
      },
      yAxis: { type: 'value', min: 0, max: 100, axisLabel: { formatter: '{value}%' } },
      series: [
        { name: 'Positivo', type: 'bar', data: posData, itemStyle: { color: '#2e7d32' } },
        { name: 'Neutral',  type: 'bar', data: neuData, itemStyle: { color: '#f9a825' } },
        { name: 'Negativo', type: 'bar', data: negData, itemStyle: { color: '#c62828' } }
      ]
    };
  }
}
