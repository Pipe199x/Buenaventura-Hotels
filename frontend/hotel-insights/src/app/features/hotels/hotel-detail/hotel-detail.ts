import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute } from '@angular/router';

@Component({
  selector: 'app-hotel-detail',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './hotel-detail.html',
  styleUrl: './hotel-detail.scss',
})
export class HotelDetail {
  hotelSlug = '';

  constructor(private route: ActivatedRoute) {
    this.hotelSlug = this.route.snapshot.paramMap.get('hotelSlug') ?? '';
  }
}
