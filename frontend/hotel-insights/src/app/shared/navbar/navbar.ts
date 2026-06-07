import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterLink, RouterLinkActive } from '@angular/router';
import { AuthService } from '../../core/auth.service';
import { AuthModalService } from '../auth-modal/auth-modal.service';

@Component({
  selector: 'app-navbar',
  standalone: true,
  imports: [CommonModule, RouterLink, RouterLinkActive],
  templateUrl: './navbar.html',
  styleUrl: './navbar.scss',
})
export class Navbar {
  private auth = inject(AuthService);
  private router = inject(Router);
  private authModal = inject(AuthModalService);

  readonly session$ = this.auth.session$;

  openLogin() {
    this.authModal.open('login');
  }

  async logout() {
    await this.auth.signOut();
    this.router.navigateByUrl('/');
  }
}
