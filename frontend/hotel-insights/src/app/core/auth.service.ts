import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import type { Session, AuthChangeEvent } from '@supabase/supabase-js';
import { supabase } from './supabase.client';

@Injectable({ providedIn: 'root' })
export class AuthService {
  private _session$ = new BehaviorSubject<Session | null>(null);
  session$ = this._session$.asObservable();

  async init() {
    const { data, error } = await supabase.auth.getSession();
    if (error) console.error('getSession error:', error.message);
    this._session$.next(data.session);

    supabase.auth.onAuthStateChange(
      (_event: AuthChangeEvent, session: Session | null) => {
        this._session$.next(session);
      }
    );
  }

  signInWithGoogle() {
    return supabase.auth.signInWithOAuth({
      provider: 'google',
      options: { redirectTo: window.location.origin }
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
