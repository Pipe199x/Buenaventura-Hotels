import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink, ActivatedRoute } from '@angular/router';
import { Subscription } from 'rxjs';

import { AuthService } from '../../../core/auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink],
  templateUrl: './login.html',
  styleUrl: './login.scss',
})
export class Login implements OnInit, OnDestroy {
  email = '';
  password = '';
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

    // Handle OAuth return: if session exists, continue to the target page.
    this.sessionSub = this.auth.session$.subscribe((session) => {
      if (session) {
        this.router.navigateByUrl(this.redirectUrl);
      }
    });
  }

  ngOnDestroy(): void {
    this.sessionSub?.unsubscribe();
  }

  async loginGoogle() {
    const redirectTo = `${window.location.origin}/login?redirect=${encodeURIComponent(this.redirectUrl)}`;

    const { error } = await this.auth.signInWithGoogle(redirectTo);

    if (error) {
      alert(error.message);
    }
  }

  async loginEmail() {
    const { error } = await this.auth.signIn(this.email, this.password);

    if (error) {
      alert(error.message);
      return;
    }

    this.router.navigateByUrl(this.redirectUrl);
  }
}
