import { Component, OnInit } from '@angular/core';
import { Router, RouterLink, ActivatedRoute } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { AuthService } from '../../../core/auth.service';

@Component({
  selector: 'app-register',
  standalone: true,
  imports: [RouterLink, FormsModule],
  templateUrl: './register.html',
  styleUrl: './register.scss',
})
export class Register implements OnInit {
  email = '';
  password = '';
  confirmPassword = '';

  // mostrar contraseña
  showPassword = false;

  // ⬅️ NUEVO: a dónde redirigir después del registro
  redirectUrl = '/home';

  constructor(
    private auth: AuthService,
    private router: Router,
    private route: ActivatedRoute
  ) {}

  ngOnInit(): void {
    // leer query param redirect
    const redirect = this.route.snapshot.queryParamMap.get('redirect');

    if (redirect) {
      this.redirectUrl = redirect;
    }
  }

  async register() {
    if (this.password !== this.confirmPassword) {
      alert('Las contraseñas no coinciden.');
      return;
    }

    const { error } = await this.auth.signUp(this.email, this.password);

    if (error) {
      alert(error.message);
      return;
    }

    alert('Cuenta creada. Revisa tu correo si se requiere confirmación.');

    // ⬅️ REDIRECCIÓN INTELIGENTE
    this.router.navigateByUrl(this.redirectUrl);
  }

  // login con Google
  async registerGoogle() {
    const { error } = await this.auth.signInWithGoogle();
    if (error) alert(error.message);
  }
}