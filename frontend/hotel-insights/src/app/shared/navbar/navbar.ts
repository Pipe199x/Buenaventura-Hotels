import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterLink, RouterLinkActive } from '@angular/router';
import { TranslateModule, TranslateService } from '@ngx-translate/core';

import { AuthService } from '../../core/auth.service';

@Component({
  selector: 'app-navbar',
  standalone: true,
  imports: [CommonModule, RouterLink, RouterLinkActive, TranslateModule],
  templateUrl: './navbar.html',
  styleUrl: './navbar.scss',
})
export class Navbar {
  // safer: define after constructor runs
  get currentLang() {
    return this.translate.currentLang || 'es';
  }

  constructor(
    private auth: AuthService,
    private router: Router,
    private translate: TranslateService
  ) {}

  setLang(lang: 'es' | 'en') {
    localStorage.setItem('lang', lang);
    this.translate.use(lang);
  }

  async logout() {
    await this.auth.signOut();
    this.router.navigateByUrl('/login');
  }
}
