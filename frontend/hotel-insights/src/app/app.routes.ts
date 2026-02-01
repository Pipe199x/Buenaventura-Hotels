import { Routes } from '@angular/router';
import { ShellComponent } from './shared/shell/shell';

// Auth
import { Login } from './features/auth/login/login';
import { Register } from './features/auth/register/register';

// Pages
import { Home } from './features/home/home';
import { Hotels } from './features/hotels/hotels';
import { About } from './features/about/about';

export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'home' },

  // Auth
  { path: 'login', component: Login },
  { path: 'register', component: Register },

  // App (con navbar)
  {
    path: '',
    component: ShellComponent,
    children: [
      { path: 'home', component: Home },
      { path: 'hotels', component: Hotels },
      { path: 'about', component: About },
    ],
  },

  { path: '**', redirectTo: 'home' },
];
