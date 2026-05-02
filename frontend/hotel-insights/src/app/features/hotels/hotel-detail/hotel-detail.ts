import {
  Component,
  OnInit,
  ChangeDetectorRef,
  NgZone,
  inject,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute } from '@angular/router';

import { supabase } from '../../../core/supabase/supabase.client';

import { SentimentTrendLine } from '../../../shared/charts/sentiment-trend-line/sentiment-trend-line';
import { ResponseRateDonut } from '../../../shared/charts/response-rate-donut/response-rate-donut';

import {
  HomeDataService,
  HotelResponseRateRow,
} from '../../../core/data/home-data.service';

type BadgeTone = 'positive' | 'neutral' | 'negative';

type SatisfactionBadge = {
  pct: number;
  tone: BadgeTone;
  label: 'Positiva' | 'Neutral' | 'Negativa';
};

@Component({
  selector: 'app-hotel-detail',
  standalone: true,
  imports: [CommonModule, SentimentTrendLine, ResponseRateDonut],
  templateUrl: './hotel-detail.html',
  styleUrl: './hotel-detail.scss',
})
export class HotelDetail implements OnInit {
  hotelSlug = '';
  hotel: any = null;

  loading = true;
  error = false;

  trendRows: any[] = [];
  responseRate: HotelResponseRateRow | null = null;

  badge: SatisfactionBadge = {
    pct: 0,
    tone: 'neutral',
    label: 'Neutral',
  };

  private route = inject(ActivatedRoute);
  private cdr = inject(ChangeDetectorRef);
  private zone = inject(NgZone);
  private homeData = inject(HomeDataService);

  ngOnInit(): void {
    this.route.paramMap.subscribe((params) => {
      this.hotelSlug = params.get('hotelSlug') ?? '';
      this.loadHotel();
    });
  }

  async loadHotel(): Promise<void> {
    this.loading = true;
    this.error = false;
    this.hotel = null;
    this.trendRows = [];
    this.responseRate = null;

    try {
      const { data, error } = await supabase
        .from('vw_home_hotels_cards_count')
        .select('*')
        .eq('hotel_name', this.hotelSlug)
        .single();

      if (error) throw error;

      const trend = await this.homeData.getSentimentTrendByHotel(data.hotel_name);
      const responseRate = await this.homeData.getHotelResponseRateByHotel(data.hotel_name);

      this.zone.run(() => {
        this.hotel = data;
        this.trendRows = trend;
        this.responseRate = responseRate;

        this.computeBadge(data?.satisfaction_rate_hotel);

        this.loading = false;
        this.error = false;

        this.cdr.detectChanges();
      });
    } catch (err) {
      console.error('Error cargando hotel:', err);

      this.zone.run(() => {
        this.error = true;
        this.loading = false;

        this.cdr.detectChanges();
      });
    }
  }

  private computeBadge(pctRaw: number | null | undefined): void {
    const pct = Math.round(Number(pctRaw ?? 0));

    if (pct >= 63) {
      this.badge = {
        pct,
        tone: 'positive',
        label: 'Positiva',
      };
    } else if (pct >= 60) {
      this.badge = {
        pct,
        tone: 'neutral',
        label: 'Neutral',
      };
    } else {
      this.badge = {
        pct,
        tone: 'negative',
        label: 'Negativa',
      };
    }
  }
}
