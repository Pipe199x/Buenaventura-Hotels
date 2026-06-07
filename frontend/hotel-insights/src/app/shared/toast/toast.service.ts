import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

export type ToastType = 'error' | 'success' | 'info';

export interface ToastMessage {
  id: number;
  type: ToastType;
  text: string;
}

@Injectable({ providedIn: 'root' })
export class ToastService {
  private _toasts$ = new BehaviorSubject<ToastMessage[]>([]);
  readonly toasts$ = this._toasts$.asObservable();

  private nextId = 1;

  show(text: string, type: ToastType = 'info', durationMs = 4000): void {
    const id = this.nextId++;
    this._toasts$.next([...this._toasts$.value, { id, type, text }]);

    if (durationMs > 0) {
      setTimeout(() => this.dismiss(id), durationMs);
    }
  }

  error(text: string, durationMs = 5000): void {
    this.show(text, 'error', durationMs);
  }

  success(text: string, durationMs = 4000): void {
    this.show(text, 'success', durationMs);
  }

  dismiss(id: number): void {
    this._toasts$.next(this._toasts$.value.filter((t) => t.id !== id));
  }
}
