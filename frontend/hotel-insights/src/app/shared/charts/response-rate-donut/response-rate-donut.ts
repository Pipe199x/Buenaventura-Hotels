import { Component, Input, OnChanges, PLATFORM_ID, inject } from '@angular/core';
import { CommonModule, isPlatformBrowser } from '@angular/common';

import type { EChartsOption } from 'echarts';

import { NgxEchartsDirective, provideEchartsCore } from 'ngx-echarts';

import * as echarts from 'echarts/core';
import { PieChart } from 'echarts/charts';
import { TooltipComponent, GraphicComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

echarts.use([
  PieChart,
  TooltipComponent,
  GraphicComponent,
  CanvasRenderer,
]);

type ResponseRate = {
  total_reviews: number;
  total_reviews_responded: number;
  pct_reviews_responded: number;
};

@Component({
  selector: 'app-response-rate-donut',
  standalone: true,
  // ECharts manages its own canvas DOM in the browser; skip hydration so the
  // server/client DOM never mismatches on this subtree.
  host: { ngSkipHydration: 'true' },
  imports: [CommonModule, NgxEchartsDirective],
  providers: [provideEchartsCore({ echarts })],
  templateUrl: './response-rate-donut.html',
  styleUrl: './response-rate-donut.scss',
})
export class ResponseRateDonut implements OnChanges {
  @Input() responseRate: ResponseRate | null = null;

  // ECharts (canvas) only initializes in the browser; never during SSR/prerender.
  protected readonly isBrowser = isPlatformBrowser(inject(PLATFORM_ID));

  chartOption: EChartsOption = {};

  ngOnChanges(): void {
    this.chartOption = this.buildOption(this.responseRate);
  }

  private buildOption(responseRate: ResponseRate | null): EChartsOption {
    if (!responseRate) {
      return {};
    }

    const total = Number(responseRate.total_reviews ?? 0);
    const responded = Number(responseRate.total_reviews_responded ?? 0);
    const notResponded = Math.max(total - responded, 0);

    const rawPct = Number(responseRate.pct_reviews_responded ?? 0);

    const pctLabel =
      rawPct > 0 && rawPct < 1
        ? '<1%'
        : `${Math.round(rawPct)}%`;

    return {
      tooltip: {
        trigger: 'item',
        formatter: '{b}: {c}',
      },

      graphic: [
        {
          type: 'text',
          left: 'center',
          top: '38%',
          style: {
            text: pctLabel,
            fontSize: 36,
            fontWeight: 700,
            fill: '#111827',
          },
        },
        {
          type: 'text',
          left: 'center',
          top: '52%',
          style: {
            text: 'RESPONDIDAS',
            fontSize: 12,
            fontWeight: 600,
            fill: '#64748b',
          },
        },
      ],

      series: [
        {
          name: 'Reseñas',
          type: 'pie',
          radius: ['58%', '78%'],
          center: ['50%', '45%'],
          avoidLabelOverlap: false,
          label: { show: false },
          labelLine: { show: false },
          stillShowZeroSum: true,
          data: [
            {
              value: responded,
              name: 'Respondidas',
              itemStyle: { color: '#3157d9' },
            },
            {
              value: notResponded,
              name: 'Sin responder',
              itemStyle: { color: '#dbe3ef' },
            },
          ],
        },
      ],
    };
  }
}
