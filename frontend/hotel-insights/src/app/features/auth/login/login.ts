import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';

import { TranslateModule } from '@ngx-translate/core';

import { AuthService } from '../../../core/auth/auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink, TranslateModule],
  templateUrl: './login.html',
  styleUrl: './login.scss',
})
export class Login {
  email = '';
  password = '';
  showPassword = false;
  loading = false;

  constructor(private auth: AuthService, private router: Router) {}

  async loginGoogle() {
    try {
      this.loading = true;
      const { error } = await this.auth.signInWithGoogle();
      if (error) alert(error.message);
    } finally {
      this.loading = false;
    }
  }

  async loginEmail() {
    try {
      this.loading = true;
      const { error } = await this.auth.signIn(this.email, this.password);
      if (error) alert(error.message);
      else this.router.navigateByUrl('/');
    } finally {
      this.loading = false;
    }
  }
}
