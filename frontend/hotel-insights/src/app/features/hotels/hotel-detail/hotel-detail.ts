import {
  Component,
  OnInit,
  ChangeDetectorRef,
  NgZone,
  inject,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute } from '@angular/router';
import { Title, Meta } from '@angular/platform-browser';

import { supabase } from '../../../core/supabase/supabase.client';

import { SchemaService } from '../../../core/seo/schema.service';
import { SITE_ORIGIN, getHotelMeta, HotelMeta } from '../../../core/seo/hotels.metadata';
import { buildHotelSchema, buildBreadcrumb } from '../../../core/seo/hotel-schema';

import { SentimentTrendLine } from '../../../shared/charts/sentiment-trend-line/sentiment-trend-line';
import { ResponseRateDonut } from '../../../shared/charts/response-rate-donut/response-rate-donut';
import { NegativeTopicsTable } from '../../../shared/charts/negative-topics-table/negative-topics-table';

import {
  HomeDataService,
  HotelResponseRateRow,
  NegativeTopicRow,
  SentimentTrendRow,
} from '../../../core/data/home-data.service';
import { PrerenderStateService } from '../../../core/data/prerender-state.service';

// Serializable per-hotel snapshot baked at build time and transferred to the client.
type HotelSnapshot = {
  hotel: any;
  trendRows: SentimentTrendRow[];
  responseRate: HotelResponseRateRow | null;
  negativeTopics: NegativeTopicRow[];
};

type BadgeTone = 'positive' | 'neutral' | 'negative';

type SatisfactionBadge = {
  pct: number;
  tone: BadgeTone;
  label: 'Positiva' | 'Neutral' | 'Negativa';
};

@Component({
  selector: 'app-hotel-detail',
  standalone: true,
  imports: [
    CommonModule,
    SentimentTrendLine,
    ResponseRateDonut,
    NegativeTopicsTable,
  ],
  templateUrl: './hotel-detail.html',
  styleUrl: './hotel-detail.scss',
})
export class HotelDetail implements OnInit {
  hotelSlug = '';
  // Display name resolved synchronously from the slug (available during prerender,
  // before the async hotel data loads) so the <h1> is crawlable.
  hotelName = '';
  hotel: any = null;

  loading = true;
  error = false;

  trendRows: any[] = [];
  responseRate: HotelResponseRateRow | null = null;
  negativeTopics: NegativeTopicRow[] = [];

  badge: SatisfactionBadge = {
    pct: 0,
    tone: 'neutral',
    label: 'Neutral',
  };

  private route = inject(ActivatedRoute);
  private cdr = inject(ChangeDetectorRef);
  private zone = inject(NgZone);
  private homeData = inject(HomeDataService);
  private prerender = inject(PrerenderStateService);
  private schemaService = inject(SchemaService);
  private title = inject(Title);
  private meta = inject(Meta);

  ngOnInit(): void {
    this.route.paramMap.subscribe((params) => {
      this.hotelSlug = params.get('hotelSlug') ?? '';
      // Base schema + title/meta are set synchronously so they are captured during
      // prerender (Supabase data loads async, outside the zone, and may not be).
      this.setBaseSeo(this.hotelSlug);

      // On the client's initial load, reuse this hotel's data baked at build time
      // (consumed synchronously so the first render matches the server — no refetch).
      const snapshot = this.prerender.take<HotelSnapshot>('hotel:' + this.hotelSlug);
      if (snapshot) {
        this.applyHotel(this.hotelSlug, snapshot);
      } else {
        this.loadHotel();
      }
    });
  }

  // Applies a resolved/transferred snapshot to the view state.
  private applyHotel(slug: string, snap: HotelSnapshot): void {
    this.hotel = snap.hotel;
    this.trendRows = snap.trendRows ?? [];
    this.responseRate = snap.responseRate ?? null;
    this.negativeTopics = snap.negativeTopics ?? [];

    this.computeBadge(this.hotel?.satisfaction_rate_hotel);

    // Re-set the Hotel schema enriched with address + aggregateRating.
    const meta = getHotelMeta(slug) ?? {
      slug,
      displayName: this.hotel?.hotel_display_name ?? slug,
    };
    this.setHotelSchema(meta, this.hotel);

    this.error = !this.hotel;
    this.loading = false;
  }

  // Fetches every per-hotel view and returns a serializable snapshot.
  private async fetchHotel(slug: string): Promise<HotelSnapshot> {
    const { data, error } = await supabase
      .from('vw_home_hotels_cards_count')
      .select('*')
      .eq('hotel_name', slug)
      .single();

    if (error) throw error;

    const trendRows = await this.homeData.getSentimentTrendByHotel(data.hotel_name);
    const responseRate = await this.homeData.getHotelResponseRateByHotel(data.hotel_name);
    const negativeTopics = await this.homeData.getNegativeTopicsByHotel(data.hotel_name);

    return { hotel: data, trendRows, responseRate, negativeTopics };
  }

  // Hotel + breadcrumb schema with static fields only (no rating/address yet).
  private setBaseSeo(slug: string): void {
    const meta = getHotelMeta(slug);
    if (!meta) return;

    this.hotelName = meta.displayName;

    this.setHotelSchema(meta);

    this.title.setTitle(`${meta.displayName} — Reseñas y análisis | Buenaventura Datos`);
    this.meta.updateTag({
      name: 'description',
      content: `Análisis de reseñas, calificación y tendencias de sentimiento de ${meta.displayName} en Buenaventura.`,
    });
  }

  // Sets (or re-sets) the Hotel + BreadcrumbList JSON-LD. `data` enriches the Hotel
  // with PostalAddress + AggregateRating once it is loaded.
  private setHotelSchema(meta: HotelMeta, data?: any): void {
    this.schemaService.setSchema('schema-hotel-detail', {
      '@context': 'https://schema.org',
      ...buildHotelSchema(meta, data),
    });

    this.schemaService.setSchema(
      'schema-hotel-breadcrumb',
      buildBreadcrumb([
        { name: 'Inicio', url: `${SITE_ORIGIN}/` },
        { name: 'Hoteles', url: `${SITE_ORIGIN}/hotels` },
        { name: meta.displayName, url: `${SITE_ORIGIN}/hotels/${meta.slug}` },
      ])
    );
  }

  async loadHotel(): Promise<void> {
    const slug = this.hotelSlug;

    this.loading = true;
    this.error = false;
    this.hotel = null;
    this.trendRows = [];
    this.responseRate = null;
    this.negativeTopics = [];

    try {
      // resolve() blocks prerender stability until the fetch + render complete and, on
      // the server, stores the snapshot in TransferState for the client to reuse.
      await this.prerender.resolve(
        'hotel:' + slug,
        () => this.fetchHotel(slug),
        (snap) => {
          this.zone.run(() => {
            this.applyHotel(slug, snap);
            this.cdr.detectChanges();
          });
        }
      );
    } catch (err) {
      console.error('Error loading hotel:', err);

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
