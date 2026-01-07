import { Routes } from '@angular/router';

import { Home } from './features/home/home';
import { About } from './features/about/about';

import { Hotels } from './features/hotels/hotels';
import { Login } from './features/auth/login/login';
import { Register } from './features/auth/register/register';

export const routes: Routes = [
  { path: '', component: Home },
  { path: 'about', component: About },
  { path: 'hotels', component: Hotels },
  { path: 'login', component: Login },
  { path: 'register', component: Register },
  { path: '**', redirectTo: '' }
];
