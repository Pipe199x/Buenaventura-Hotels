import { Component, OnInit, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';

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

  async ngOnInit() {
    this.loading = true;
    this.errorMsg = '';

    try {
      this.hotels = await this.homeData.getHotelCards();
      this.hotelBadge = this.buildBadgesFromSatisfaction(this.hotels);
    } catch (e: any) {
      this.errorMsg = e?.message ?? 'Error cargando hoteles';
    } finally {
      this.loading = false;
    }
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
