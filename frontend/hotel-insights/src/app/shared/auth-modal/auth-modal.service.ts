import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

export type AuthMode = 'login' | 'register';

@Injectable({ providedIn: 'root' })
export class AuthModalService {
  private _isOpen$ = new BehaviorSubject<boolean>(false);
  readonly isOpen$ = this._isOpen$.asObservable();

  private _mode$ = new BehaviorSubject<AuthMode>('login');
  readonly mode$ = this._mode$.asObservable();

  /** Where to navigate after a successful auth (null = stay on current page). */
  redirectUrl: string | null = null;

  open(mode: AuthMode = 'login', redirectUrl: string | null = null): void {
    this.redirectUrl = redirectUrl;
    this._mode$.next(mode);
    this._isOpen$.next(true);
  }

  close(): void {
    this._isOpen$.next(false);
    this.redirectUrl = null;
  }

  setMode(mode: AuthMode): void {
    this._mode$.next(mode);
  }
}
