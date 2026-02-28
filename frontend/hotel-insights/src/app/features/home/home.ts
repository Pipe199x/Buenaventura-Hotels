import {
  Component,
  OnInit,
  ChangeDetectorRef,
  inject,
  DestroyRef,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, NavigationEnd, RouterLink } from '@angular/router';
import { filter } from 'rxjs/operators';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

import {
  HomeDataService,
  HomeOverviewGlobal,
  HotelCardRow,
  SentimentDistributionRow,
} from '../../core/data/home-data.service';

import { SentimentStackedBarComponent } from '../../shared/charts/sentiment-stacked-bar/sentiment-stacked-bar';

type BadgeTone = 'positive' | 'neutral' | 'negative';

type HotelBadge = {
  pct: number;
  tone: BadgeTone;
  label: 'Positiva' | 'Neutral' | 'Negativa';
};

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [CommonModule, RouterLink, SentimentStackedBarComponent],
  templateUrl: './home.html',
  styleUrl: './home.scss',
})
export class Home implements OnInit {
  loading = true;
  chartLoading = true;
  errorMsg = '';

  kpis: HomeOverviewGlobal | null = null;
  hotels: HotelCardRow[] = [];
  sentimentRows: SentimentDistributionRow[] = [];

  hotelBadge = new Map<string, HotelBadge>();

  private homeData = inject(HomeDataService);
  private router = inject(Router);
  private cdr = inject(ChangeDetectorRef);

  // ✅ CLAVE: esto arregla el NG0203
  private destroyRef = inject(DestroyRef);

  private inFlight = false;

  ngOnInit(): void {
    // 1) Carga inicial
    this.loadHome('init');

    // 2) Cada vez que se navega a /home, recarga (sin depender de doble clic)
    this.router.events
      .pipe(
        filter((e): e is NavigationEnd => e instanceof NavigationEnd),
        filter(
          (e) =>
            e.urlAfterRedirects === '/home' ||
            e.urlAfterRedirects.startsWith('/home')
        ),
        // ✅ FIX REAL: pasar DestroyRef
        takeUntilDestroyed(this.destroyRef)
      )
      .subscribe(() => {
        this.loadHome('nav');
      });
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

  getBadgeForHotel(hotelName: string): HotelBadge {
    return (
      this.hotelBadge.get(hotelName) ?? {
        pct: 0,
        tone: 'neutral',
        label: 'Neutral',
      }
    );
  }

  private async loadHome(reason: 'init' | 'nav'): Promise<void> {
    // Evita que 2 recargas se pisen (muy común con router events)
    if (this.inFlight) return;
    this.inFlight = true;

    // DEBUG visible en consola (sin warnings de label duplicado)
    console.groupCollapsed(`[Home] loadHome(${reason})`);
    const t0 = performance.now();

    this.loading = true;
    this.chartLoading = true;
    this.errorMsg = '';

    // Fuerza que el spinner aparezca INMEDIATO (sin esperar otro click)
    this.cdr.detectChanges();

    try {
      // Timeout para evitar “loading infinito” si algo queda colgado
      const withTimeout = async <T>(
        p: Promise<T>,
        ms: number,
        label: string
      ): Promise<T> => {
        let t: any;
        const timeout = new Promise<never>((_, rej) => {
          t = setTimeout(
            () => rej(new Error(`Timeout (${ms}ms) en ${label}`)),
            ms
          );
        });
        const out = await Promise.race([p, timeout]);
        clearTimeout(t);
        return out as T;
      };

      const results = await Promise.allSettled([
        withTimeout(
          this.homeData.getHomeOverviewGlobal(),
          12000,
          'getHomeOverviewGlobal'
        ),
        withTimeout(this.homeData.getHotelCards(), 12000, 'getHotelCards'),
        withTimeout(
          this.homeData.getSentimentDistribution(),
          12000,
          'getSentimentDistribution'
        ),
      ]);

      const errs: string[] = [];

      // KPIs
      const kpisR = results[0];
      if (kpisR.status === 'fulfilled') {
        this.kpis = kpisR.value;
      } else {
        this.kpis = null;
        errs.push(kpisR.reason?.message ?? 'Error en KPIs');
      }

      // Hotels
      const hotelsR = results[1];
      if (hotelsR.status === 'fulfilled') {
        this.hotels = hotelsR.value ?? [];
      } else {
        this.hotels = [];
        errs.push(hotelsR.reason?.message ?? 'Error en hoteles');
      }

      // ✅ NUEVO: badges salen de satisfaction_rate_hotel (cards), no del chart
      // (esto es lo que necesitas para la regla 63/60)
      this.hotelBadge = this.buildBadgesFromSatisfaction(this.hotels);

      // Distribution (chart)
      const distR = results[2];
      if (distR.status === 'fulfilled') {
        this.sentimentRows = distR.value ?? [];
      } else {
        this.sentimentRows = [];
        errs.push(
          distR.reason?.message ?? 'Error en distribución de sentimientos'
        );
      }

      // Si hubo errores parciales, muéstralos (pero NO congeles la UI)
      if (errs.length) {
        this.errorMsg = errs.join(' | ');
      }

      console.log('[Home] kpis:', this.kpis);
      console.log('[Home] hotels:', this.hotels.length);
      console.log('[Home] sentimentRows:', this.sentimentRows.length);
    } catch (e: any) {
      this.errorMsg = e?.message ?? 'Error cargando datos del Home';
      console.error('[Home] load error:', e);
    } finally {
      this.loading = false;
      this.chartLoading = false;

      // CLAVE: fuerza repaint al terminar async (evita “doble clic para que aparezca”)
      this.cdr.detectChanges();

      const ms = Math.round(performance.now() - t0);
      console.log(`[Home] total: ${ms}ms`);
      console.groupEnd();

      this.inFlight = false;
    }
  }

  // ✅ BADGES por satisfacción (regla 63/60)
  private buildBadgesFromSatisfaction(hotels: HotelCardRow[]): Map<string, HotelBadge> {
    const out = new Map<string, HotelBadge>();

    for (const h of hotels) {
      const pct = Math.round(Number(h.satisfaction_rate_hotel ?? 0));

      let tone: BadgeTone;
      let label: HotelBadge['label'];

      if (pct >= 63) {
        tone = 'positive';
        label = 'Positiva';
      } else if (pct >= 60) {
        tone = 'neutral';
        label = 'Neutral';
      } else {
        tone = 'negative';
        label = 'Negativa';
      }

      out.set(h.hotel_name, { pct, tone, label });
    }

    return out;
  }
}
