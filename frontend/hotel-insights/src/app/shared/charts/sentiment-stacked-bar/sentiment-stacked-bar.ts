import { Component, Input, OnChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import type { EChartsOption } from 'echarts';
import { NgxEchartsDirective, provideEchartsCore } from 'ngx-echarts';

// ECharts core
import * as echarts from 'echarts/core';
import { BarChart } from 'echarts/charts';
import {
  TooltipComponent,
  GridComponent,
  LegendComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

echarts.use([
  BarChart,
  TooltipComponent,
  GridComponent,
  LegendComponent,
  CanvasRenderer,
]);

type Row = {
  hotel_display_name: string;
  sentiment_label: 'positive' | 'neutral' | 'negative';
  pct_label: number;
};

type SeriesName = 'Positivo' | 'Neutral' | 'Negativo';

@Component({
  selector: 'app-sentiment-stacked-bar',
  standalone: true,
  imports: [CommonModule, NgxEchartsDirective],
  providers: [provideEchartsCore({ echarts })],
  template: `
    <div
      echarts
      class="chart"
      [options]="option"
      (chartInit)="onChartInit($event)"
      (chartClick)="onChartClick($event)">
    </div>
  `,
  styles: [
    `
      .chart {
        width: 100%;
        height: 420px;
      }
    `,
  ],
})
export class SentimentStackedBarComponent implements OnChanges {
  @Input({ required: true }) rows: Row[] = [];

  option: EChartsOption = {};

  private chart: any | null = null;

  ngOnChanges(): void {
    if (!this.rows || !this.rows.length) {
      this.option = {};
      return;
    }
    this.option = this.buildOption(this.rows);
  }

  onChartInit(ec: any) {
    this.chart = ec;
  }

  onChartClick(e: any) {
    // Click en una barra -> alternar (toggle) su serie
    // e.seriesName será 'Positivo' | 'Neutral' | 'Negativo'
    const seriesName: SeriesName | undefined = e?.seriesName;
    if (!seriesName || !this.chart) return;

    // Toggle programático de legend
    this.chart.dispatchAction({
      type: 'legendToggleSelect',
      name: seriesName,
    });
  }

  private buildOption(rows: Row[]): EChartsOption {
    const hotels = Array.from(new Set(rows.map(r => r.hotel_display_name)));

    const getPct = (hotel: string, label: Row['sentiment_label']) => {
      const found = rows.find(
        r => r.hotel_display_name === hotel && r.sentiment_label === label
      );
      return found?.pct_label ?? 0;
    };

    const positive = hotels.map(h => getPct(h, 'positive'));
    const neutral  = hotels.map(h => getPct(h, 'neutral'));
    const negative = hotels.map(h => getPct(h, 'negative'));

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: ((params: any) => {
          const hotel = params?.[0]?.axisValue ?? '';

          const pos = params.find((p: any) => p.seriesName === 'Positivo')?.value ?? 0;
          const neu = params.find((p: any) => p.seriesName === 'Neutral')?.value ?? 0;
          const neg = params.find((p: any) => p.seriesName === 'Negativo')?.value ?? 0;

          return `
            <div style="font-weight:700;margin-bottom:6px;">${hotel}</div>
            <div>🟩 Positivo: <b>${pos}%</b></div>
            <div>🟨 Neutral: <b>${neu}%</b></div>
            <div>🟥 Negativo: <b>${neg}%</b></div>
          `;
        }) as any,
      },

      legend: {
        // ✅ en la mitad del chart (arriba-centro)
        left: 'center',
        top: 8,
        data: ['Positivo', 'Neutral', 'Negativo'],

        // ✅ comportamiento de click en legend (toggle)
        // 'multiple' deja apagar/encender varias
        // 'single' dejaría solo una activa a la vez
        selectedMode: 'multiple',

        itemWidth: 10,
        itemHeight: 10,
        textStyle: { fontSize: 12 },
      },

      grid: {
        left: 80,
        right: 32,
        top: 52,     // 👈 más espacio porque legend ahora está centrada
        bottom: 100,
        containLabel: false,
      },

      xAxis: {
        type: 'category',
        data: hotels,
        name: 'Hoteles analizados',
        nameLocation: 'middle',
        nameGap: 70,
        nameTextStyle: {
          fontSize: 13,
          fontWeight: 700,
          color: '#1f2937',
        },
        axisLabel: {
          margin: 18,
          fontSize: 11,
          color: '#374151',
          rotate: 0,
        },
        axisTick: { alignWithLabel: true },
      },

      yAxis: {
        type: 'value',
        min: 0,
        max: 100,
        name: 'Porcentaje (%)',
        nameLocation: 'middle',
        nameGap: 55,
        nameRotate: 90,
        nameTextStyle: {
          fontSize: 12,
          fontWeight: 600,
          color: '#374151',
        },
        axisLabel: {
          formatter: '{value}%',
          fontSize: 11,
        },
      },

      series: [
        {
          name: 'Positivo',
          type: 'bar',
          data: positive,
          itemStyle: { color: '#2e7d32' },
          barGap: '10%',
          barCategoryGap: '35%',
          emphasis: { focus: 'series' },
        },
        {
          name: 'Neutral',
          type: 'bar',
          data: neutral,
          itemStyle: { color: '#f9a825' },
          emphasis: { focus: 'series' },
        },
        {
          name: 'Negativo',
          type: 'bar',
          data: negative,
          itemStyle: { color: '#c62828' },
          emphasis: { focus: 'series' },
        },
      ],
    };
  }
}
