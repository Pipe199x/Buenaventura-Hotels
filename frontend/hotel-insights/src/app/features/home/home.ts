import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HomeDataService, HomeOverviewGlobal } from '../../core/data/home-data.service';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './home.html',
  styleUrl: './home.scss',
})
export class Home implements OnInit {
  loading = true;
  errorMsg = '';
  kpis: HomeOverviewGlobal | null = null;

  constructor(private homeData: HomeDataService) {}

  async ngOnInit() {
    try {
      this.kpis = await this.homeData.getHomeOverviewGlobal();
    } catch (e: any) {
      this.errorMsg = e?.message ?? 'Error cargando KPIs';
    } finally {
      this.loading = false;
    }
  }
}
