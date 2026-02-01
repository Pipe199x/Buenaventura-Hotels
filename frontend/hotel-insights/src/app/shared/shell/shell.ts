import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { Navbar } from '../navbar/navbar';

@Component({
  selector: 'app-shell',
  standalone: true,
  imports: [Navbar, RouterOutlet],
  template: `
    <app-navbar></app-navbar>
    <main class="shell-main">
      <router-outlet></router-outlet>
    </main>
  `,
  styles: [
    `
      .shell-main {
        padding: 24px;
        max-width: 1200px;
        margin: 0 auto;
      }
    `,
  ],
})
export class ShellComponent {}
