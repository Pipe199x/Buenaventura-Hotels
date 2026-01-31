import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterLink, RouterLinkActive } from '@angular/router';
import { AuthService } from '../../core/auth.service';
import { Observable } from 'rxjs';

@Component({
  selector: 'app-navbar',
  standalone: true,
  imports: [CommonModule, RouterLink, RouterLinkActive],
  templateUrl: './navbar.html',
  styleUrl: './navbar.scss', // o navbar.scss si usas scss
})
export class Navbar {
  session$: Observable<any>;

  constructor(private auth: AuthService, private router: Router) {
    this.session$ = this.auth.session$;
  }

  async logout() {
    await this.auth.signOut();
    this.router.navigateByUrl('/login');
  }
}
