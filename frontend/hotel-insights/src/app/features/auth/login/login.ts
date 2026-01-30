import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';

import { AuthService } from '../../../core/auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink],
  templateUrl: './login.html',
  styleUrl: './login.scss',
})
export class Login {
  email = '';
  password = '';
  showPassword = false;

  constructor(private auth: AuthService, private router: Router) {}

  async loginGoogle() {
    const { error } = await this.auth.signInWithGoogle();
    if (error) alert(error.message);
  }

  async loginEmail() {
    const { error } = await this.auth.signIn(this.email, this.password);
    if (error) alert(error.message);
    else this.router.navigateByUrl('/');
  }
}
