import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';

import { ToastComponent } from './shared/toast/toast';
import { AuthModalComponent } from './shared/auth-modal/auth-modal';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, ToastComponent, AuthModalComponent],
  template: `
    <router-outlet></router-outlet>
    <app-toast />
    <app-auth-modal />
  `,
})
export class AppComponent {}
