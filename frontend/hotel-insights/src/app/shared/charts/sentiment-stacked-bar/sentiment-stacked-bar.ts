import { Component, Input, OnChanges } from '@angular/core';
import { CommonModule } from '@angular/common';

import type { EChartsOption } from 'echarts';
import type { TopLevelFormatterParams } from 'echarts/types/dist/shared';

import { NgxEchartsDirective, provideEchartsCore } from 'ngx-echarts';

import * as echarts from 'echarts/core';
import { BarChart } from 'echarts/charts';
import { TooltipComponent, GridComponent, LegendComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

echarts.use([BarChart, TooltipComponent, GridComponent, LegendComponent, CanvasRenderer]);

type Row = {
  hotel_display_name: string;
  sentiment_label: 'positive' | 'neutral' | 'negative';
  pct_label: number;
};

@Component({
  selector: 'app-sentiment-stacked-bar',
  standalone: true,
  // ECharts manages its own canvas DOM in the browser; skip hydration so the
  // server/client DOM never mismatches on this subtree.
  host: { ngSkipHydration: 'true' },
  imports: [CommonModule, NgxEchartsDirective],
  providers: [provideEchartsCore({ echarts })],
  template: `<div echarts class="echart" [options]="option"></div>`,
  styles: [
    `
      .echart { width: 100%; height: 360px; }
      @media (max-width: 420px) { .echart { height: 360px; } }
    `,
  ],
})
export class SentimentStackedBarComponent implements OnChanges {
  @Input({ required: true }) rows: Row[] = [];

  option: EChartsOption = {};

  ngOnChanges(): void {
    this.option = this.buildOption(this.rows);
  }

  private buildOption(rows: Row[]): EChartsOption {
    const hotelsRaw = Array.from(new Set(rows.map((r) => r.hotel_display_name)));

    // Common name prefixes removed for shorter axis labels.
    const shortLabel = (name: string) => {
      let s = (name ?? '').trim();
      s = s.replace(/^(hotel|hostal)\s+/i, '').trim();
      s = s.replace(/\s+/g, ' ').trim();
      return s || name;
    };

    const hotelsShort = hotelsRaw.map((h) => shortLabel(h));

    const getPct = (hotelRaw: string, label: Row['sentiment_label']) => {
      const found = rows.find(
        (r) => r.hotel_display_name === hotelRaw && r.sentiment_label === label
      );
      return found?.pct_label ?? 0;
    };

    const pos = hotelsRaw.map((h) => getPct(h, 'positive'));
    const neu = hotelsRaw.map((h) => getPct(h, 'neutral'));
    const neg = hotelsRaw.map((h) => getPct(h, 'negative'));

    // Max value rounded to the next 10 and clamped to [50, 100].
    const computedMax = (() => {
      const all = [...pos, ...neu, ...neg].map((n) => Number(n) || 0);
      const top = Math.max(0, ...all);
      const rounded = Math.ceil(top / 10) * 10; // 71.2 -> 80
      return Math.min(100, Math.max(50, rounded));
    })();

    // Shared axis max for desktop and mobile configs.
    const axisMax = computedMax;

    const tooltip = {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: (params: TopLevelFormatterParams) => {
        const arr = Array.isArray(params) ? params : [params];
        const axisValue = (arr[0] as any)?.axisValue ?? '';
        const lines = arr
          .map((p: any) => `${p.marker} ${p.seriesName}: ${p.value}%`)
          .join('<br/>');
        return `<div style="font-weight:700;margin-bottom:4px;">${axisValue}</div>${lines}`;
      },
    } as const;

    // Desktop layout: grouped vertical bars.
    const desktop: EChartsOption = {
      tooltip,

      legend: {
        top: 10,
        left: 'center',
        selectedMode: true,
        itemWidth: 12,
        itemHeight: 12,
        textStyle: { fontSize: 12 },
      },

      grid: {
        left: 56,
        right: 20,
        top: 52,
        bottom: 72,
        containLabel: true,
      },

      xAxis: {
        type: 'category',
        data: hotelsRaw,
        name: 'Hoteles analizados',
        nameLocation: 'middle',
        nameGap: 46,
        axisLabel: {
          interval: 0,
          rotate: 0,
          hideOverlap: true,
        },
      },

      yAxis: {
        type: 'value',
        min: 0,
        max: axisMax, // Dynamic y-axis max.
        name: 'Porcentaje (%)',
        nameLocation: 'middle',
        nameGap: 50,
        axisLabel: { formatter: '{value}%' },
      },

      series: [
        { name: 'Positivo', type: 'bar', data: pos, itemStyle: { color: '#2e7d32' }, barMaxWidth: 26 },
        { name: 'Neutral',  type: 'bar', data: neu, itemStyle: { color: '#f9a825' }, barMaxWidth: 26 },
        { name: 'Negativo', type: 'bar', data: neg, itemStyle: { color: '#c62828' }, barMaxWidth: 26 },
      ],
    };

    // Mobile layout: grouped horizontal bars.
    const mobile: EChartsOption = {
      tooltip,

      legend: {
        top: 10,
        left: 'center',
        selectedMode: true,
        itemWidth: 10,
        itemHeight: 10,
        textStyle: { fontSize: 11 },
      },

      grid: {
        left: 118,
        right: 0,
        top: 46,
        bottom: 26,
        containLabel: false,
      },

      // Percentage x-axis.
      xAxis: {
        type: 'value',
        min: 0,
        max: axisMax, // Dynamic x-axis max.
        splitNumber: axisMax <= 60 ? 3 : 4, // Tick density is reduced on narrow screens.
        name: 'Porcentaje (%)',
        nameLocation: 'middle',
        nameGap: 36,
        axisLabel: {
          fontSize: 10,
          margin: 10,
          // Labels displayed at 20-point steps.
          formatter: (value: number) => {
            const v = Math.round(Number(value));
            return v % 20 === 0 ? `${v}%` : '';
          },
          showMinLabel: true,
          showMaxLabel: true,
        },
      },

      // Hotel label axis.
      yAxis: {
        type: 'category',
        data: hotelsShort,
        name: 'Hoteles analizados',
        nameLocation: 'middle',
        nameGap: 72,
        axisLabel: {
          fontSize: 11,
          margin: 10,
          formatter: (value: string) => (value.length > 16 ? value.slice(0, 16) + '…' : value),
        },
      },

      series: [
        { name: 'Positivo', type: 'bar', data: pos, itemStyle: { color: '#2e7d32' }, barMaxWidth: 12 },
        { name: 'Neutral',  type: 'bar', data: neu, itemStyle: { color: '#f9a825' }, barMaxWidth: 12 },
        { name: 'Negativo', type: 'bar', data: neg, itemStyle: { color: '#c62828' }, barMaxWidth: 12 },
      ],
    };

    return {
      ...desktop,
      media: [{ query: { maxWidth: 420 }, option: mobile }],
    };
  }
}
