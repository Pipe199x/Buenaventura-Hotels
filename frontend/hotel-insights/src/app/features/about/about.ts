import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';

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
export class About {

  objetivos: ObjetivoCard[] = [
    {
      titulo: 'Identificar fortalezas',
      descripcion: 'Detectar aspectos positivos valorados por los huéspedes.',
      icono: 'assets/strength.png',
    },
    {
      titulo: 'Áreas de mejora',
      descripcion: 'Señalar oportunidades de mejora en los servicios hoteleros.',
      icono: 'assets/areas de mejora.png',
    },
    {
      titulo: 'Tendencias temporales',
      descripcion: 'Analizar cómo evoluciona la percepción de los visitantes a lo largo del tiempo.',
      icono: 'assets/temporal trends.png',
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
