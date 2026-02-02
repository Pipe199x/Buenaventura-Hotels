import { Component, Input, OnChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import type { EChartsOption } from 'echarts';
import { NgxEchartsDirective, provideEchartsCore } from 'ngx-echarts';
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

@Component({
  selector: 'app-sentiment-stacked-bar',
  standalone: true,
  imports: [CommonModule, NgxEchartsDirective],
  providers: [provideEchartsCore({ echarts })],
  templateUrl: './sentiment-stacked-bar.html',
  styleUrl: './sentiment-stacked-bar.scss',
})
export class SentimentStackedBarComponent implements OnChanges {
  @Input({ required: true }) rows: Row[] = [];

  option: EChartsOption = {};

  ngOnChanges(): void {
    this.option = this.buildOption(this.rows);
  }

  private buildOption(rows: Row[]): EChartsOption {
    // 1) hoteles únicos en el orden que venga (no te importa el orden)
    const hotels = Array.from(
      new Set(rows.map((r) => r.hotel_display_name))
    );

    // 2) helper para obtener % por hotel y label (default 0)
    const getPct = (hotel: string, label: Row['sentiment_label']) => {
      const found = rows.find(
        (r) => r.hotel_display_name === hotel && r.sentiment_label === label
      );
      return found?.pct_label ?? 0;
    };

    const positive = hotels.map((h) => getPct(h, 'positive'));
    const neutral = hotels.map((h) => getPct(h, 'neutral'));
    const negative = hotels.map((h) => getPct(h, 'negative'));

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params: any) => {
          // params = array de series para ese hotel
          const hotel = params?.[0]?.axisValue ?? '';
          const p = params.find((x: any) => x.seriesName === 'Positivo')?.value ?? 0;
          const n = params.find((x: any) => x.seriesName === 'Neutral')?.value ?? 0;
          const ne = params.find((x: any) => x.seriesName === 'Negativo')?.value ?? 0;

          return `
            <div style="font-weight:600;margin-bottom:4px;">${hotel}</div>
            <div>Positivo: ${p}%</div>
            <div>Neutral: ${n}%</div>
            <div>Negativo: ${ne}%</div>
          `;
        },
      },
      legend: {
        top: 0,
      },
      grid: {
        left: 24,
        right: 24,
        top: 40,
        bottom: 24,
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: hotels,
        axisLabel: { rotate: 0 },
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 100,
        axisLabel: { formatter: '{value}%' },
      },
      series: [
        {
          name: 'Positivo',
          type: 'bar',
          stack: 'total',
          emphasis: { focus: 'series' },
          data: positive,
        },
        {
          name: 'Neutral',
          type: 'bar',
          stack: 'total',
          emphasis: { focus: 'series' },
          data: neutral,
        },
        {
          name: 'Negativo',
          type: 'bar',
          stack: 'total',
          emphasis: { focus: 'series' },
          data: negative,
        },
      ],
    };
  }
}
