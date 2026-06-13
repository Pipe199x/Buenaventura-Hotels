import {
  Component,
  OnInit,
  ChangeDetectorRef,
  inject,
  DestroyRef,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, NavigationEnd } from '@angular/router';
import { Title, Meta } from '@angular/platform-browser';
import { filter } from 'rxjs/operators';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

import { HomeDataService, HotelCardRow } from '../../../core/data/home-data.service';
import { SchemaService } from '../../../core/seo/schema.service';
import { CanonicalService } from '../../../core/seo/canonical.service';
import { SITE_ORIGIN, HOTELS } from '../../../core/seo/hotels.metadata';
import { buildHotelSchema } from '../../../core/seo/hotel-schema';
import { AuthModalService } from '../../../shared/auth-modal/auth-modal.service';
import { AuthService } from '../../../core/auth.service';
import { PrerenderStateService } from '../../../core/data/prerender-state.service';

type BadgeTone = 'positive' | 'neutral' | 'negative';

type HotelBadge = {
  pct: number;
  tone: BadgeTone;
  label: 'Positiva' | 'Neutral' | 'Negativa';
};

@Component({
  selector: 'app-hotels-list',
  standalone: true,
  imports: [CommonModule],
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
  private schemaService = inject(SchemaService);
  private canonical = inject(CanonicalService);
  private authModal = inject(AuthModalService);
  private auth = inject(AuthService);
  private prerender = inject(PrerenderStateService);
  private title = inject(Title);
  private meta = inject(Meta);

  private inFlight = false;
  // Set when hydrated from a build-time snapshot, to skip the initial refetch.
  private skipNextReload = false;

  ngOnInit(): void {
    this.setPageMeta();
    this.setHotelsCollectionSchema();

    // Reuse the data baked at build time on the client's initial load (no refetch).
    const snapshot = this.prerender.take<HotelCardRow[]>('hotels-list');
    if (snapshot) {
      this.applyHotels(snapshot);
      this.skipNextReload = true;
    } else {
      this.loadHotels('init');
    }

    this.router.events
      .pipe(
        filter((e): e is NavigationEnd => e instanceof NavigationEnd),
        filter((e) => e.urlAfterRedirects === '/hotels'),
        takeUntilDestroyed(this.destroyRef)
      )
      .subscribe(() => {
        this.setPageMeta();
        this.setHotelsCollectionSchema();
        if (this.skipNextReload) {
          this.skipNextReload = false;
          return;
        }
        this.loadHotels('nav');
      });
  }

  private setPageMeta(): void {
    this.title.setTitle('Hoteles analizados en Buenaventura | Buenaventura Datos');
    this.canonical.setCanonical(`${SITE_ORIGIN}/hotels`);
    this.meta.updateTag({
      name: 'description',
      content:
        'Listado de hoteles de Buenaventura analizados mediante reseñas de Google: calificaciones, análisis de sentimientos y tendencias de percepción turística.',
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

  // If already authenticated, go straight to the hotel detail; otherwise prompt to
  // sign in (register tab) and route there after a successful auth.
  openDetails(hotelName: string): void {
    const target = '/hotels/' + hotelName;
    if (this.auth.currentSession) {
      this.router.navigateByUrl(target);
    } else {
      this.authModal.open('register', target);
    }
  }

  // Builds the CollectionPage + ItemList schema from the shared HOTELS metadata.
  // Called synchronously (no rows -> base Hotel items, captured at prerender) and
  // again after load with `rows` so each ListItem carries an AggregateRating.
  private setHotelsCollectionSchema(rows?: HotelCardRow[]): void {
    const byName = new Map((rows ?? []).map((r) => [r.hotel_name, r]));

    const itemListElement = HOTELS.map((meta, i) => ({
      '@type': 'ListItem',
      position: i + 1,
      item: buildHotelSchema(meta, byName.get(meta.slug)),
    }));

    this.schemaService.setSchema('schema-hotels-collection-page', {
      '@context': 'https://schema.org',
      '@type': 'CollectionPage',
      '@id': `${SITE_ORIGIN}/hotels#collection`,
      name: 'Hoteles analizados en Buenaventura',
      url: `${SITE_ORIGIN}/hotels`,
      inLanguage: 'es-CO',
      description:
        'Listado de hoteles de Buenaventura analizados mediante reseñas de Google, análisis de sentimientos, temáticas, calificaciones y tendencias de percepción turística.',
      isPartOf: {
        '@id': `${SITE_ORIGIN}/#website`,
      },
      about: {
        '@id': `${SITE_ORIGIN}/#dataset`,
      },
      mainEntity: {
        '@type': 'ItemList',
        name: 'Hoteles analizados en Buenaventura',
        itemListElement,
      },
    });
  }

  // Applies a resolved/transferred snapshot to the view state.
  private applyHotels(hotels: HotelCardRow[]): void {
    this.hotels = hotels ?? [];
    this.hotelBadge = this.buildBadgesFromSatisfaction(this.hotels);
    // Enrich the ItemList with per-hotel AggregateRating now that data is available.
    this.setHotelsCollectionSchema(this.hotels);
    this.loading = false;
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

      // resolve() blocks prerender stability until the fetch + render complete and, on
      // the server, stores the snapshot in TransferState for the client to reuse.
      await this.prerender.resolve(
        'hotels-list',
        () => withTimeout(this.homeData.getHotelCards(), 12000, `getHotelCards:${reason}`),
        (hotels) => {
          this.applyHotels(hotels);
          this.cdr.detectChanges();
        }
      );
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

  private buildBadgesFromSatisfaction(
    hotels: HotelCardRow[]
  ): Map<string, HotelBadge> {
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
