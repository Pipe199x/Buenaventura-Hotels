import { Component, OnInit, ChangeDetectorRef, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, NavigationEnd } from '@angular/router';
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
  imports: [CommonModule, SentimentStackedBarComponent],
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

  private inFlight = false;

  ngOnInit(): void {
    // 1) Carga inicial
    this.loadHome('init');

    // 2) Cada vez que se navega a /home, recarga (sin depender de doble clic)
    this.router.events
      .pipe(
        filter((e): e is NavigationEnd => e instanceof NavigationEnd),
        filter((e) => e.urlAfterRedirects === '/home' || e.urlAfterRedirects.startsWith('/home')),
        takeUntilDestroyed()
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

    // DEBUG visible en consola
    console.groupCollapsed(`[Home] loadHome(${reason})`);
    console.time('[Home] total');

    this.loading = true;
    this.chartLoading = true;
    this.errorMsg = '';

    // Fuerza que el spinner aparezca INMEDIATO (sin esperar otro click)
    this.cdr.detectChanges();

    try {
      // Timeout para evitar “loading infinito” si algo queda colgado
      const withTimeout = async <T>(p: Promise<T>, ms: number, label: string): Promise<T> => {
        let t: any;
        const timeout = new Promise<never>((_, rej) => {
          t = setTimeout(() => rej(new Error(`Timeout (${ms}ms) en ${label}`)), ms);
        });
        const out = await Promise.race([p, timeout]);
        clearTimeout(t);
        return out as T;
      };

      const results = await Promise.allSettled([
        withTimeout(this.homeData.getHomeOverviewGlobal(), 12000, 'getHomeOverviewGlobal'),
        withTimeout(this.homeData.getHotelCards(), 12000, 'getHotelCards'),
        withTimeout(this.homeData.getSentimentDistribution(), 12000, 'getSentimentDistribution'),
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

      // Distribution (chart)
      const distR = results[2];
      if (distR.status === 'fulfilled') {
        this.sentimentRows = distR.value ?? [];
        this.hotelBadge = this.buildBadges(this.sentimentRows);
      } else {
        this.sentimentRows = [];
        this.hotelBadge = new Map();
        errs.push(distR.reason?.message ?? 'Error en distribución de sentimientos');
      }

      // Si hubo errores parciales, muéstralos (pero NO congeles la UI)
      if (errs.length) {
        this.errorMsg = errs.join(' | ');
      }

      // DEBUG
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

      console.timeEnd('[Home] total');
      console.groupEnd();
      this.inFlight = false;
    }
  }

  private buildBadges(rows: SentimentDistributionRow[]): Map<string, HotelBadge> {
    const byHotel = new Map<string, SentimentDistributionRow[]>();

    for (const r of rows) {
      if (!byHotel.has(r.hotel_name)) byHotel.set(r.hotel_name, []);
      byHotel.get(r.hotel_name)!.push(r);
    }

    const out = new Map<string, HotelBadge>();

    for (const [hotelName, list] of byHotel.entries()) {
      const best = [...list].sort((a, b) => (b.pct_label ?? 0) - (a.pct_label ?? 0))[0];

      const tone = best?.sentiment_label ?? 'neutral';
      const pct = Math.round(best?.pct_label ?? 0);

      const label =
        tone === 'positive' ? 'Positiva' : tone === 'negative' ? 'Negativa' : 'Neutral';

      out.set(hotelName, { pct, tone, label });
    }

    return out;
  }
}
