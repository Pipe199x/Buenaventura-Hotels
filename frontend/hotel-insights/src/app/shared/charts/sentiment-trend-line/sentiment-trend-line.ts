import { Component, Input, OnChanges } from '@angular/core';
import { CommonModule } from '@angular/common';

import type { EChartsOption } from 'echarts';
import type { TopLevelFormatterParams } from 'echarts/types/dist/shared';

import { NgxEchartsDirective, provideEchartsCore } from 'ngx-echarts';

import * as echarts from 'echarts/core';
import { LineChart } from 'echarts/charts';
import { TooltipComponent, GridComponent, LegendComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

echarts.use([LineChart, TooltipComponent, GridComponent, LegendComponent, CanvasRenderer]);

type Row = {
  year: number;
  sentiment_label: 'positive' | 'neutral' | 'negative';
  pct_year_decimal: number;
};

@Component({
  selector: 'app-sentiment-trend-line',
  standalone: true,
  imports: [CommonModule, NgxEchartsDirective],
  providers: [provideEchartsCore({ echarts })],
  templateUrl: './sentiment-trend-line.html',
  styleUrl: './sentiment-trend-line.scss',
})
export class SentimentTrendLine implements OnChanges {
  @Input() rows: Row[] = [];

  chartOption: EChartsOption = {};

  ngOnChanges(): void {
    this.chartOption = this.buildOption(this.rows);
  }

  private buildOption(rows: Row[]): EChartsOption {
    const years = Array.from(new Set(rows.map((r) => r.year))).sort();

    const getPct = (year: number, label: Row['sentiment_label']) => {
      const found = rows.find(
        (r) => r.year === year && r.sentiment_label === label
      );

      return Math.round(Number(found?.pct_year_decimal ?? 0) * 10000) / 100;
    };

    const pos = years.map((y) => getPct(y, 'positive'));
    const neu = years.map((y) => getPct(y, 'neutral'));
    const neg = years.map((y) => getPct(y, 'negative'));

    const tooltip = {
      trigger: 'axis',
      formatter: (params: TopLevelFormatterParams) => {
        const arr = Array.isArray(params) ? params : [params];
        const axisValue = (arr[0] as any)?.axisValue ?? '';

        const lines = arr
          .map((p: any) => `${p.marker} ${p.seriesName}: ${p.value}%`)
          .join('<br/>');

        return `<div style="font-weight:700;margin-bottom:4px;">${axisValue}</div>${lines}`;
      },
    } as const;

    return {
      tooltip,

      legend: {
        top: 6,
        left: 'center',
        itemWidth: 12,
        itemHeight: 12,
        textStyle: {
          fontSize: 12,
          fontWeight: 400,
        },
      },

      // ✅ MAIN FIX → MORE WIDTH
      grid: {
        left: 40,     // ⬅️ smaller = more chart width
        right: 6,
        top: 40,
        bottom: 45,
        containLabel: false, // ⬅️ critical
      },

      xAxis: {
        type: 'category',
        data: years,
        name: 'Periodo',
        nameLocation: 'middle',
        nameGap: 28,
        nameTextStyle: {
          fontSize: 13,
          fontWeight: 400, // not bold
          color: '#374151',
        },
        axisLabel: {
          fontSize: 11,
          color: '#374151',
        },
      },

      yAxis: {
        type: 'value',
        min: 0,
        max: 100,
        name: 'Porcentaje (%)',
        nameLocation: 'middle',
        nameGap: 34,
        nameTextStyle: {
          fontSize: 13,
          fontWeight: 400, // not bold
          color: '#374151',
        },
        axisLabel: {
          formatter: '{value}%',
          fontSize: 11,
          color: '#374151',
          margin: 6,
        },
      },

      series: [
        {
          name: 'Positivo',
          type: 'line',
          data: pos,
          smooth: true,
          symbol: 'circle',
          symbolSize: 7,
          lineStyle: { width: 3, color: '#2e7d32' },
          itemStyle: { color: '#2e7d32' },
        },
        {
          name: 'Neutral',
          type: 'line',
          data: neu,
          smooth: true,
          symbol: 'circle',
          symbolSize: 7,
          lineStyle: { width: 3, color: '#f9a825' },
          itemStyle: { color: '#f9a825' },
        },
        {
          name: 'Negativo',
          type: 'line',
          data: neg,
          smooth: true,
          symbol: 'circle',
          symbolSize: 7,
          lineStyle: { width: 3, color: '#c62828' },
          itemStyle: { color: '#c62828' },
        },
      ],
    };
  }
}
