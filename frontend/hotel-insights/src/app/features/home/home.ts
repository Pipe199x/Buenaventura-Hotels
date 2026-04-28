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

  // DestroyRef instance used by takeUntilDestroyed.
  private destroyRef = inject(DestroyRef);

  private inFlight = false;

  ngOnInit(): void {
    // Initial data load.
    this.loadHome('init');

    // Data reload on route revisit.
    this.router.events
      .pipe(
        filter((e): e is NavigationEnd => e instanceof NavigationEnd),
        filter(
          (e) =>
            e.urlAfterRedirects === '/' ||
            e.urlAfterRedirects.startsWith('/')
        ),
        // Stream lifecycle bound to component destroy.
        takeUntilDestroyed(this.destroyRef)
      )
      .subscribe(() => {
        this.loadHome('nav');
      });
  }

  // Comma-separated words transformed into chip items.
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
    // Overlapping reload requests are skipped.
    if (this.inFlight) return;
    this.inFlight = true;

    // Grouped debug logs per request cycle.
    console.groupCollapsed(`[Home] loadHome(${reason})`);
    const t0 = performance.now();

    this.loading = true;
    this.chartLoading = true;
    this.errorMsg = '';

    // UI state flushed before async calls.
    this.cdr.detectChanges();

    try {
      // Per-query timeout wrapper.
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

      // KPIs.
      const kpisR = results[0];
      if (kpisR.status === 'fulfilled') {
        this.kpis = kpisR.value;
      } else {
        this.kpis = null;
        errs.push(kpisR.reason?.message ?? 'Error en KPIs');
      }

      // Hotel cards.
      const hotelsR = results[1];
      if (hotelsR.status === 'fulfilled') {
        this.hotels = hotelsR.value ?? [];
      } else {
        this.hotels = [];
        errs.push(hotelsR.reason?.message ?? 'Error en hoteles');
      }

      // Badge color/label derived from satisfaction_rate_hotel.
      this.hotelBadge = this.buildBadgesFromSatisfaction(this.hotels);

      // Chart distribution rows.
      const distR = results[2];
      if (distR.status === 'fulfilled') {
        this.sentimentRows = distR.value ?? [];
      } else {
        this.sentimentRows = [];
        errs.push(
          distR.reason?.message ?? 'Error en distribución de sentimientos'
        );
      }

      // Partial errors are displayed while available data still renders.
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

      // Final UI state flush after async completion.
      this.cdr.detectChanges();

      const ms = Math.round(performance.now() - t0);
      console.log(`[Home] total: ${ms}ms`);
      console.groupEnd();

      this.inFlight = false;
    }
  }

  // Satisfaction thresholds: >=63 positive, >=60 neutral, otherwise negative.
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
