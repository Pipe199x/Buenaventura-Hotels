import { ApplicationConfig, APP_INITIALIZER } from '@angular/core';
import { provideRouter, withRouterConfig } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';

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
        // ✅ si haces click en Inicio estando en /home, recarga el componente
        onSameUrlNavigation: 'reload',
      })
    ),

    // ✅ CLAVE: NO withFetch()
    provideHttpClient(),

    // ✅ ECharts global
    provideEchartsCore({ echarts }),

    // ✅ Auth init
    {
      provide: APP_INITIALIZER,
      useFactory: initAuth,
      deps: [AuthService],
      multi: true,
    },
  ],
};
