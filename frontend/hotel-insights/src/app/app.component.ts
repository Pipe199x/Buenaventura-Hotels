import { Component, OnInit } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { Navbar } from './shared/navbar/navbar';
import { AuthService } from './core/auth.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, Navbar],
  template: `
    <app-navbar></app-navbar>
    <main class="app">
      <router-outlet></router-outlet>
    </main>
  `,
  styles: [`
    .app { padding: 18px; max-width: 1200px; margin: 0 auto; }
  `]
})
export class AppComponent implements OnInit {
  constructor(private auth: AuthService) {}

  ngOnInit() {
    this.auth.init();
  }
}
