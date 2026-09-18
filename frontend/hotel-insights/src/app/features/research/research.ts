import { ChangeDetectorRef, Component, OnInit, PLATFORM_ID, inject } from '@angular/core';
import { CommonModule, isPlatformBrowser } from '@angular/common';
import { RouterLink } from '@angular/router';
import { Title, Meta } from '@angular/platform-browser';

import { SchemaService } from '../../core/seo/schema.service';
import { CanonicalService } from '../../core/seo/canonical.service';
import { RESEARCH, SITE_ORIGIN } from '../../core/seo/hotels.metadata';
import { buildBreadcrumb } from '../../core/seo/hotel-schema';

type Cifra = {
  valor: string;
  etiqueta: string;
};

type FichaItem = {
  campo: string;
  valor: string;
};

// Published KPIs, taken from section 7.2 of the thesis (figures 14 to 18).
// These are the study's own numbers, not live Supabase data: they are the
// snapshot the published document reports, so they stay fixed here.
type ResultadoHotel = {
  slug: string;
  nombre: string;
  resenas: string;
  puntuacion: string;
  positivas: string;
  negativas: string;
  neutras: string;
  respondidas: string;
};

const PAGE_URL = `${SITE_ORIGIN}/investigacion`;

const DESCRIPTION =
  'Trabajo de grado que analizó 3.727 reseñas de Google sobre los cinco hoteles más ' +
  'comentados de Buenaventura entre 2020 y 2025. Resultados por hotel, validación del ' +
  'modelo, documento completo. Universidad de Manizales, 2026.';

@Component({
  selector: 'app-research',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './research.html',
  styleUrl: './research.scss',
})
export class Research implements OnInit {
  private schemaService = inject(SchemaService);
  private canonical = inject(CanonicalService);
  private title = inject(Title);
  private meta = inject(Meta);
  private cdr = inject(ChangeDetectorRef);

  protected readonly isBrowser = isPlatformBrowser(inject(PLATFORM_ID));

  research = RESEARCH;

  // The inline viewer starts closed, so the prerendered HTML carries no iframe and
  // the 1,38 MB PDF is only fetched once somebody asks for it.
  viewerOpen = false;

  toggleViewer(): void {
    if (!this.isBrowser) return;

    // ponytail: heurística simple. El visor de PDF embebido no es fiable en móvil
    // (iOS Safari suele mostrar solo la primera página), así que ahí se abre aparte.
    // Si algún día hace falta más precisión, aquí es donde se afina.
    const puedeIncrustar = navigator.pdfViewerEnabled !== false && window.innerWidth >= 900;

    if (!puedeIncrustar) {
      window.open(RESEARCH.localPdfPath, '_blank', 'noopener');
      return;
    }

    this.viewerOpen = !this.viewerOpen;

    // The app runs zoneless (no zone.js, no polyfills entry in angular.json), so a
    // plain property write renders nothing on its own. Same pattern as home.ts:249
    // and hotel-detail.ts:198.
    this.cdr.detectChanges();

    // Reading mode hides everything above the viewer, so the document ends up at the
    // top of the page: scroll there instead of chasing the element.
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  ngOnInit(): void {
    this.title.setTitle(
      'Investigación: análisis de sentimientos en reseñas hoteleras de Buenaventura | Buenaventura Datos'
    );
    // Canonical points at this page, never at the repository record: the text
    // here is original, not a copy of the RIDUM abstract.
    this.canonical.setCanonical(PAGE_URL);
    this.meta.updateTag({ name: 'description', content: DESCRIPTION });

    this.schemaService.setSchema('schema-research-article', {
      '@context': 'https://schema.org',
      '@type': 'ScholarlyArticle',
      '@id': `${PAGE_URL}#thesis`,
      name: RESEARCH.title,
      headline: RESEARCH.title,
      url: PAGE_URL,
      sameAs: [RESEARCH.handleUrl],
      datePublished: RESEARCH.year,
      inLanguage: 'es-CO',
      description: DESCRIPTION,
      author: { '@id': `${SITE_ORIGIN}/#author` },
      publisher: {
        '@type': 'CollegeOrUniversity',
        name: 'Universidad de Manizales',
        url: 'https://umanizales.edu.co',
      },
      license: RESEARCH.licenseUrl,
      isPartOf: { '@id': `${SITE_ORIGIN}/#website` },
      about: { '@id': `${SITE_ORIGIN}/#dataset` },
      keywords: [
        'análisis de sentimientos',
        'minería de texto',
        'reseñas hoteleras',
        'percepción turística',
        'reputación digital',
        'hoteles en Buenaventura',
      ],
      encoding: {
        '@type': 'MediaObject',
        contentUrl: RESEARCH.pdfUrl,
        encodingFormat: 'application/pdf',
      },
      citation:
        'Duque Caicedo, A. F. (2026). Análisis de sentimientos en reseñas hoteleras de ' +
        'Buenaventura mediante minería de texto [Trabajo de grado profesional]. ' +
        'Universidad de Manizales. RIDUM: Repositorio Institucional Universidad de Manizales.',
    });

    this.schemaService.setSchema(
      'schema-research-breadcrumb',
      buildBreadcrumb([
        { name: 'Inicio', url: `${SITE_ORIGIN}/` },
        { name: 'Investigación', url: PAGE_URL },
      ])
    );
  }

  cifras: Cifra[] = [
    { valor: '3.727', etiqueta: 'reseñas analizadas' },
    { valor: '1.710', etiqueta: 'de ellas con texto, base del análisis de sentimientos' },
    { valor: '2020 - 2025', etiqueta: 'periodo analizado' },
    { valor: '90,2 %', etiqueta: 'de concordancia con la revisión humana' },
  ];

  resultados: ResultadoHotel[] = [
    {
      slug: 'cordillera',
      nombre: 'Hotel Cordillera',
      resenas: '1.193',
      puntuacion: '3,94',
      positivas: '47,48 %',
      negativas: '34,53 %',
      neutras: '17,99 %',
      respondidas: '0,92 %',
    },
    {
      slug: 'cosmos_pacifico',
      nombre: 'Hotel Cosmos Pacífico',
      resenas: '788',
      puntuacion: '4,46',
      positivas: '63,37 %',
      negativas: '27,72 %',
      neutras: '8,91 %',
      respondidas: '0 %',
    },
    {
      slug: 'maguipi',
      nombre: 'Hotel Magüipí',
      resenas: '765',
      puntuacion: '4,70',
      positivas: '69,66 %',
      negativas: '12,36 %',
      neutras: '17,98 %',
      respondidas: '57,25 %',
    },
    {
      slug: 'torre_mar',
      nombre: 'Hotel Torre Mar',
      resenas: '575',
      puntuacion: '4,57',
      positivas: '70,51 %',
      negativas: '20,51 %',
      neutras: '8,97 %',
      respondidas: '0 %',
    },
    {
      slug: 'steven_buenaventura',
      nombre: 'Hotel Steven Buenaventura',
      resenas: '406',
      puntuacion: '4,17',
      positivas: '62,34 %',
      negativas: '31,17 %',
      neutras: '6,49 %',
      respondidas: '6,70 %',
    },
  ];

  ficha: FichaItem[] = [
    { campo: 'Tipo de documento', valor: 'Trabajo de grado profesional' },
    { campo: 'Institución', valor: 'Universidad de Manizales' },
    { campo: 'Facultad', valor: 'Ciencias e Ingeniería' },
    { campo: 'Programa', valor: 'Ingeniería en Analítica de Datos' },
    { campo: 'Año', valor: '2026' },
    { campo: 'Autor', valor: 'Andrés Felipe Duque Caicedo' },
    { campo: 'Director', valor: 'Andrés Alberto Osorio Londoño, PhD' },
    { campo: 'Codirector', valor: 'Edgar Rafael Jiménez López, MSc' },
    { campo: 'Licencia', valor: 'Creative Commons BY-NC-ND 4.0' },
  ];
}
