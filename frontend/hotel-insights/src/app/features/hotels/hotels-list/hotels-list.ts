import {
  Component,
  OnInit,
  ChangeDetectorRef,
  inject,
  DestroyRef,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, NavigationEnd, RouterModule } from '@angular/router';
import { filter } from 'rxjs/operators';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

import { HomeDataService, HotelCardRow } from '../../../core/data/home-data.service';

type BadgeTone = 'positive' | 'neutral' | 'negative';

type HotelBadge = {
  pct: number;
  tone: BadgeTone;
  label: 'Positiva' | 'Neutral' | 'Negativa';
};

@Component({
  selector: 'app-hotels-list',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './hotels-list.html',
  styleUrl: './hotels-list.scss',
})
export class HotelsList implements OnInit {
  loading = true;
  errorMsg = '';
  hotels: HotelCardRow[] = [];
  hotelBadge = new Map<string, HotelBadge>();

  private homeData = inject(HomeDataService);
  private router = inject(Router);
  private cdr = inject(ChangeDetectorRef);
  private destroyRef = inject(DestroyRef);

  private inFlight = false;

  ngOnInit(): void {
    this.loadHotels('init');

    this.router.events
      .pipe(
        filter((e): e is NavigationEnd => e instanceof NavigationEnd),
        filter((e) => e.urlAfterRedirects === '/hotels'),
        takeUntilDestroyed(this.destroyRef)
      )
      .subscribe(() => {
        this.loadHotels('nav');
      });
  }

  trackByHotelName = (_: number, h: HotelCardRow) => h.hotel_name;

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

  private async loadHotels(reason: 'init' | 'nav'): Promise<void> {
    if (this.inFlight) return;
    this.inFlight = true;

    this.loading = true;
    this.errorMsg = '';
    this.cdr.detectChanges();

    try {
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

      this.hotels = await withTimeout(
        this.homeData.getHotelCards(),
        12000,
        `getHotelCards:${reason}`
      );
      this.hotelBadge = this.buildBadgesFromSatisfaction(this.hotels);
    } catch (e: any) {
      this.hotels = [];
      this.hotelBadge = new Map();
      this.errorMsg = e?.message ?? 'Error cargando hoteles';
    } finally {
      this.loading = false;
      this.inFlight = false;
      this.cdr.detectChanges();
    }
  }

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
