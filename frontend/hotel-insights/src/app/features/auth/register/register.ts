import { Component, OnInit, OnDestroy } from '@angular/core';
import { Router, RouterLink, ActivatedRoute } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';
import { AuthService } from '../../../core/auth.service';

@Component({
  selector: 'app-register',
  standalone: true,
  imports: [RouterLink, FormsModule],
  templateUrl: './register.html',
  styleUrl: './register.scss',
})
export class Register implements OnInit, OnDestroy {
  email = '';
  password = '';
  confirmPassword = '';

  showPassword = false;
  redirectUrl = '/home';
  private sessionSub?: Subscription;

  constructor(
    private auth: AuthService,
    private router: Router,
    private route: ActivatedRoute
  ) {}

  ngOnInit(): void {
    const redirect = this.route.snapshot.queryParamMap.get('redirect');

    if (redirect) {
      this.redirectUrl = redirect;
    }

    // ✅ Si vuelve de Google y ya hay sesión, redirigir automáticamente
    this.sessionSub = this.auth.session$.subscribe((session) => {
      if (session) {
        this.router.navigateByUrl(this.redirectUrl);
      }
    });
  }

  ngOnDestroy(): void {
    this.sessionSub?.unsubscribe();
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
    this.router.navigateByUrl(this.redirectUrl);
  }

  async registerGoogle() {
    const redirectTo = `${window.location.origin}/register?redirect=${encodeURIComponent(this.redirectUrl)}`;

    const { error } = await this.auth.signInWithGoogle(redirectTo);

    if (error) {
      alert(error.message);
    }
  }
}