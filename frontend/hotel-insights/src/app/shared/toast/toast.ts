import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';

import { ToastService } from './toast.service';

@Component({
  selector: 'app-toast',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './toast.html',
  styleUrl: './toast.scss',
})
export class ToastComponent {
  constructor(public toast: ToastService) {}

  dismiss(id: number): void {
    this.toast.dismiss(id);
  }
}
