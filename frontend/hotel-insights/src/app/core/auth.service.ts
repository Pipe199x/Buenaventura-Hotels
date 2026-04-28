import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import type { Session, AuthChangeEvent } from '@supabase/supabase-js';
import { supabase } from './supabase/supabase.client';

@Injectable({ providedIn: 'root' })
export class AuthService {
  private _session$ = new BehaviorSubject<Session | null>(null);
  readonly session$ = this._session$.asObservable();

  // Prevent duplicate auth listeners.
  private initialized = false;

  async init(): Promise<void> {
    if (this.initialized) return;
    this.initialized = true;

    const { data, error } = await supabase.auth.getSession();
    if (error) console.error('getSession error:', error.message);

    this._session$.next(data.session ?? null);

    supabase.auth.onAuthStateChange(
      (_event: AuthChangeEvent, session: Session | null) => {
        this._session$.next(session);
      }
    );
  }

  signInWithGoogle(redirectTo?: string) {
    return supabase.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: redirectTo ?? window.location.origin,
      },
    });
  }

  signUp(email: string, password: string) {
    return supabase.auth.signUp({ email, password });
  }

  signIn(email: string, password: string) {
    return supabase.auth.signInWithPassword({ email, password });
  }

  signOut() {
    return supabase.auth.signOut();
  }
}
