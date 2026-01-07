import { Component } from '@angular/core';
import { Router, RouterLink } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { AuthService } from '../../../core/auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [RouterLink, FormsModule],
  templateUrl: './login.html',
  styleUrl: './login.scss',
})
export class Login {
  email = '';
  password = '';

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
