import { Component, OnInit, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';

import { HomeDataService, HotelCardRow } from '../../../core/data/home-data.service';

@Component({
  selector: 'app-hotels-list',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './hotels-list.html',
  styleUrls: ['./hotels-list.scss'],
})
export class HotelsList implements OnInit {
  loading = true;
  errorMsg = '';
  hotels: HotelCardRow[] = [];

  private homeData = inject(HomeDataService);

  async ngOnInit() {
    this.loading = true;
    this.errorMsg = '';

    try {
      this.hotels = await this.homeData.getHotelCards();
    } catch (e: any) {
      this.errorMsg = e?.message ?? 'Error cargando hoteles';
    } finally {
      this.loading = false;
    }
  }

  // slug estable y escalable: hotel_name
  trackByHotelName = (_: number, h: HotelCardRow) => h.hotel_name;
}
