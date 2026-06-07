import {
  Component,
  OnInit,
  ChangeDetectorRef,
  inject,
  DestroyRef,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, NavigationEnd } from '@angular/router';
import { filter } from 'rxjs/operators';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

import { HomeDataService, HotelCardRow } from '../../../core/data/home-data.service';
import { SchemaService } from '../../../core/seo/schema.service';
import { AuthModalService } from '../../../shared/auth-modal/auth-modal.service';

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
  private authModal = inject(AuthModalService);

  private inFlight = false;

  ngOnInit(): void {
    this.setHotelsCollectionSchema();
    this.loadHotels('init');

    this.router.events
      .pipe(
        filter((e): e is NavigationEnd => e instanceof NavigationEnd),
        filter((e) => e.urlAfterRedirects === '/hotels'),
        takeUntilDestroyed(this.destroyRef)
      )
      .subscribe(() => {
        this.setHotelsCollectionSchema();
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

  // Opens the auth modal (register tab), then routes to the hotel after auth.
  openDetails(hotelName: string): void {
    this.authModal.open('register', '/hotels/' + hotelName);
  }

  private setHotelsCollectionSchema(): void {
    this.schemaService.setSchema('schema-hotels-collection-page', {
      '@context': 'https://schema.org',
      '@type': 'CollectionPage',
      '@id': 'https://buenaventuradatos.com/hotels#collection',
      name: 'Hoteles analizados en Buenaventura',
      url: 'https://buenaventuradatos.com/hotels',
      inLanguage: 'es-CO',
      description:
        'Listado de hoteles de Buenaventura analizados mediante reseñas de Google, análisis de sentimientos, temáticas, calificaciones y tendencias de percepción turística.',
      isPartOf: {
        '@id': 'https://buenaventuradatos.com/#website',
      },
      about: {
        '@id': 'https://buenaventuradatos.com/#dataset',
      },
      mainEntity: {
        '@type': 'ItemList',
        name: 'Hoteles analizados en Buenaventura',
        itemListElement: [
          {
            '@type': 'ListItem',
            position: 1,
            name: 'Hotel Cordillera',
            url: 'https://buenaventuradatos.com/hotels/cordillera',
          },
          {
            '@type': 'ListItem',
            position: 2,
            name: 'Hotel Cosmos Pacífico',
            url: 'https://buenaventuradatos.com/hotels/cosmos_pacifico',
          },
          {
            '@type': 'ListItem',
            position: 3,
            name: 'Hotel Magüipí',
            url: 'https://buenaventuradatos.com/hotels/maguipi',
          },
          {
            '@type': 'ListItem',
            position: 4,
            name: 'Hotel Torre Mar',
            url: 'https://buenaventuradatos.com/hotels/torre_mar',
          },
          {
            '@type': 'ListItem',
            position: 5,
            name: 'Hotel Steven Buenaventura',
            url: 'https://buenaventuradatos.com/hotels/steven_buenaventura',
          },
        ],
      },
    });
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
