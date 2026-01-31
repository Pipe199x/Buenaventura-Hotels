import { Routes } from '@angular/router';

import { Login } from './features/auth/login/login';
import { Register } from './features/auth/register/register';
import { Home } from './features/home/home';
import { Hotels } from './features/hotels/hotels'; // o el nombre real: HotelWorks
import { About } from './features/about/about';

export const routes: Routes = [
  // public
  { path: 'login', component: Login },
  { path: 'register', component: Register },

  // home
  { path: '', component: Home },         // ✅ Home principal
  { path: 'home', redirectTo: '', pathMatch: 'full' },  // ✅ alias /home

  // pages
  { path: 'hotels', component: Hotels },
  { path: 'about', component: About },

  // fallback
  { path: '**', redirectTo: '' },
];
