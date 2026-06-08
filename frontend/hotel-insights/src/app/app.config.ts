import { ApplicationConfig, APP_INITIALIZER } from '@angular/core';
import { provideRouter, withRouterConfig } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { provideClientHydration, withEventReplay } from '@angular/platform-browser';

import { routes } from './app.routes';
import { AuthService } from './core/auth.service';

/* =======================
   NGX-ECHARTS (CORE)
======================= */
import { provideEchartsCore } from 'ngx-echarts';

import * as echarts from 'echarts/core';
import { BarChart } from 'echarts/charts';
import { GridComponent, TooltipComponent, LegendComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

echarts.use([BarChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer]);

export function initAuth(auth: AuthService) {
  return () => auth.init();
}

export const appConfig: ApplicationConfig = {
  providers: [
    provideRouter(
      routes,
      withRouterConfig({
        // Enables component reload when navigating to the same URL.
        onSameUrlNavigation: 'reload',
      })
    ),

    // HttpClient provider.
    provideHttpClient(),

    // Reuse the server-rendered (prerendered) DOM on the client instead of
    // re-rendering it — no flicker, no layout shift.
    provideClientHydration(withEventReplay()),

    // ECharts provider at app bootstrap.
    provideEchartsCore({ echarts }),

    // Auth initializer.
    {
      provide: APP_INITIALIZER,
      useFactory: initAuth,
      deps: [AuthService],
      multi: true,
    },
  ],
};
