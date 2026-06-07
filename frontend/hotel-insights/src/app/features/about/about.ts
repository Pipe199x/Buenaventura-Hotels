import { Component, OnInit, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Title, Meta } from '@angular/platform-browser';

import { SchemaService } from '../../core/seo/schema.service';
import { SITE_ORIGIN } from '../../core/seo/hotels.metadata';
import { buildBreadcrumb } from '../../core/seo/hotel-schema';

type ObjetivoCard = {
  titulo: string;
  descripcion: string;
  icono: string;
};

@Component({
  selector: 'app-about',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './about.html',
  styleUrl: './about.scss',
})
export class About implements OnInit {
  private schemaService = inject(SchemaService);
  private title = inject(Title);
  private meta = inject(Meta);

  ngOnInit(): void {
    this.title.setTitle('Acerca del proyecto | Buenaventura Datos');
    this.meta.updateTag({
      name: 'description',
      content:
        'Objetivos, tecnologías y técnicas detrás del análisis de sentimientos de reseñas hoteleras de Buenaventura.',
    });

    this.schemaService.setSchema('schema-about-page', {
      '@context': 'https://schema.org',
      '@type': 'AboutPage',
      '@id': `${SITE_ORIGIN}/about#webpage`,
      name: 'Acerca del proyecto',
      url: `${SITE_ORIGIN}/about`,
      inLanguage: 'es-CO',
      description:
        'Objetivos, tecnologías y técnicas detrás del análisis de sentimientos de reseñas hoteleras de Buenaventura.',
      isPartOf: { '@id': `${SITE_ORIGIN}/#website` },
      about: { '@id': `${SITE_ORIGIN}/#dataset` },
      mainEntity: { '@id': `${SITE_ORIGIN}/#author` },
    });

    this.schemaService.setSchema(
      'schema-about-breadcrumb',
      buildBreadcrumb([
        { name: 'Inicio', url: `${SITE_ORIGIN}/` },
        { name: 'Acerca de', url: `${SITE_ORIGIN}/about` },
      ])
    );
  }

  objetivos: ObjetivoCard[] = [
    {
      titulo: 'Identificar fortalezas',
      descripcion: 'Detectar aspectos positivos valorados por los huéspedes.',
      icono: 'assets/strength.png',
    },
    {
      titulo: 'Áreas de mejora',
      descripcion: 'Señalar oportunidades de mejora en los servicios hoteleros.',
      icono: 'assets/feedback.png',
    },
    {
      titulo: 'Tendencias temporales',
      descripcion: 'Analizar cómo evoluciona la percepción de los visitantes a lo largo del tiempo.',
      icono: 'assets/temporal-trends.png',
    },
    {
      titulo: 'Información para el sector turístico',
      descripcion: 'Generar información útil para apoyar el análisis del turismo local.',
      icono: 'assets/turism.png',
    },
  ];

  tecnologias: string[] = [
    'Angular',
    'TypeScript',
    'HTML',
    'SCSS',
    'Python',
    'Supabase',
    'Azure Cognitive Services',
    'Netlify',
    'Google Reviews',
  ];

  tecnicas: string[] = [
    'Análisis de sentimientos para identificar percepciones positivas, neutrales y negativas.',
    'Extracción de palabras clave y términos frecuentes en las reseñas.',
    'Identificación de temáticas positivas y negativas asociadas a la experiencia de los huéspedes.',
    'Visualización interactiva de resultados para facilitar la interpretación de los datos.',
  ];
}
