import { Component, HostListener, OnDestroy, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { Subscription } from 'rxjs';

import { AuthService } from '../../core/auth.service';
import { translateAuthError } from '../../core/supabase/auth-errors';
import { ToastService } from '../toast/toast.service';
import { AuthModalService, AuthMode } from './auth-modal.service';

const PENDING_REDIRECT_KEY = 'authRedirect';

@Component({
  selector: 'app-auth-modal',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './auth-modal.html',
  styleUrl: './auth-modal.scss',
})
export class AuthModalComponent implements OnInit, OnDestroy {
  email = '';
  password = '';
  confirmPassword = '';
  showPassword = false;

  isOpen = false;

  private openSub?: Subscription;
  private sessionSub?: Subscription;

  constructor(
    private auth: AuthService,
    public authModal: AuthModalService,
    private router: Router,
    private toast: ToastService
  ) {}

  ngOnInit(): void {
    this.openSub = this.authModal.isOpen$.subscribe((open) => {
      this.isOpen = open;
      this.toggleBodyScroll(open);
      if (open) {
        this.resetForm();
      }
    });

    // After an OAuth round-trip the app reloads; if a session is now present and we
    // stashed a destination before leaving, navigate there and clear the marker.
    this.sessionSub = this.auth.session$.subscribe((session) => {
      if (!session) return;
      const pending = sessionStorage.getItem(PENDING_REDIRECT_KEY);
      if (pending) {
        sessionStorage.removeItem(PENDING_REDIRECT_KEY);
        this.router.navigateByUrl(pending);
      }
    });
  }

  ngOnDestroy(): void {
    this.openSub?.unsubscribe();
    this.sessionSub?.unsubscribe();
    this.toggleBodyScroll(false);
  }

  switchMode(mode: AuthMode): void {
    this.authModal.setMode(mode);
  }

  close(): void {
    this.authModal.close();
  }

  @HostListener('document:keydown.escape')
  onEscape(): void {
    if (this.isOpen) this.close();
  }

  async loginEmail(): Promise<void> {
    const { error } = await this.auth.signIn(this.email, this.password);

    if (error) {
      this.toast.error(translateAuthError(error));
      return;
    }

    this.finishSuccess();
  }

  async loginGoogle(): Promise<void> {
    this.stashRedirect();
    const { error } = await this.auth.signInWithGoogle(`${window.location.origin}/`);

    if (error) {
      this.toast.error(translateAuthError(error));
    }
  }

  async register(): Promise<void> {
    if (this.password !== this.confirmPassword) {
      this.toast.error('Las contraseñas no coinciden.');
      return;
    }

    const { error } = await this.auth.signUp(this.email, this.password);

    if (error) {
      this.toast.error(translateAuthError(error));
      return;
    }

    this.toast.success('Cuenta creada. Revisa tu correo si se requiere confirmación.');
    this.finishSuccess();
  }

  async registerGoogle(): Promise<void> {
    this.stashRedirect();
    const { error } = await this.auth.signInWithGoogle(`${window.location.origin}/`);

    if (error) {
      this.toast.error(translateAuthError(error));
    }
  }

  private finishSuccess(): void {
    const redirect = this.authModal.redirectUrl;
    this.close();
    if (redirect) {
      this.router.navigateByUrl(redirect);
    }
  }

  private stashRedirect(): void {
    const target = this.authModal.redirectUrl ?? this.router.url;
    sessionStorage.setItem(PENDING_REDIRECT_KEY, target);
  }

  private resetForm(): void {
    this.email = '';
    this.password = '';
    this.confirmPassword = '';
    this.showPassword = false;
  }

  private toggleBodyScroll(lock: boolean): void {
    document.body.style.overflow = lock ? 'hidden' : '';
  }
}
