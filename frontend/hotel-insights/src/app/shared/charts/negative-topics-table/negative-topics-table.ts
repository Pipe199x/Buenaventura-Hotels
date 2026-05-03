import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

export type NegativeTopicRow = {
  hotel_name: string;
  thematic_label: string;
  y2020: number;
  y2021: number;
  y2022: number;
  y2023: number;
  y2024: number;
  y2025: number;
  total: number;
  active_years: number;
};

@Component({
  selector: 'app-negative-topics-table',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './negative-topics-table.html',
  styleUrl: './negative-topics-table.scss',
})
export class NegativeTopicsTable {
  @Input() rows: NegativeTopicRow[] = [];

  // Dynamic years definition
  years = [
    { key: 'y2020', label: '2020' },
    { key: 'y2021', label: '2021' },
    { key: 'y2022', label: '2022' },
    { key: 'y2023', label: '2023' },
    { key: 'y2024', label: '2024' },
    { key: 'y2025', label: '2025' },
  ] as const;

  // Only show years that actually have data
  get visibleYears() {
    return this.years.filter((year) =>
      this.rows.some((row) => row[year.key] > 0)
    );
  }

  // Heatmap intensity logic per row
  getIntensityClass(value: number, row: NegativeTopicRow): string {
    const values = [
      row.y2020,
      row.y2021,
      row.y2022,
      row.y2023,
      row.y2024,
      row.y2025,
    ];

    const max = Math.max(...values);

    if (value === 0 || max === 0) {
      return 'zero';
    }

    const ratio = value / max;

    if (ratio > 0.75) {
      return 'high';
    }

    if (ratio > 0.5) {
      return 'medium';
    }

    if (ratio > 0.25) {
      return 'low';
    }

    return 'very-low';
  }
}
