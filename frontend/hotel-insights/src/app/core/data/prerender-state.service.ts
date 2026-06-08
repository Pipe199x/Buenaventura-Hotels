import {
  Injectable,
  PendingTasks,
  PLATFORM_ID,
  TransferState,
  makeStateKey,
  inject,
} from '@angular/core';
import { isPlatformServer } from '@angular/common';

/**
 * Bridges build-time (prerender/SSR) data into the browser without a refetch.
 *
 * - `resolve()` runs a data fetch inside a `PendingTasks` task so prerender waits for it
 *   (Supabase resolves outside Angular's zone, so zone-based stability wouldn't see it).
 *   The result is applied to the view AND, on the server, stored in `TransferState`
 *   — all inside the task, so the task is only released after the DOM reflects the data,
 *   guaranteeing prerender serializes the populated view (not the loading state).
 * - `take()` is called synchronously on the client during `ngOnInit` to consume that
 *   transferred snapshot before the first change detection, so the hydrated DOM matches
 *   the server render and no client fetch is needed for the landing route.
 */
@Injectable({ providedIn: 'root' })
export class PrerenderStateService {
  private readonly ts = inject(TransferState);
  private readonly pending = inject(PendingTasks);
  private readonly isServer = isPlatformServer(inject(PLATFORM_ID));

  /** Synchronously consume a server-transferred snapshot (single-use), or null. */
  take<T>(key: string): T | null {
    const k = makeStateKey<T>(key);
    if (!this.ts.hasKey(k)) return null;
    const value = this.ts.get<T | null>(k, null);
    this.ts.remove(k);
    return value;
  }

  /**
   * Fetch the data, store it for transfer (server only), then apply it to the view —
   * all while a pending task keeps the app unstable, so prerender blocks until the
   * fetched data has been rendered. `apply` should set component state and flush change
   * detection.
   */
  async resolve<T>(
    key: string,
    fetcher: () => Promise<T>,
    apply: (value: T) => void
  ): Promise<void> {
    // add() registers a stability-blocking task and returns a cleanup function. The
    // task is only released in `finally`, after `apply` has rendered the data — so
    // prerender serializes the populated view, not the loading state.
    const done = this.pending.add();
    try {
      const value = await fetcher();
      if (this.isServer) {
        this.ts.set(makeStateKey<T>(key), value);
      }
      apply(value);
    } finally {
      done();
    }
  }
}
