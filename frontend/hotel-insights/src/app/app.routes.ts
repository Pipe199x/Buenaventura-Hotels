import { Routes } from '@angular/router';
import { ShellComponent } from './shared/shell/shell';

// Auth
import { Login } from './features/auth/login/login';
import { Register } from './features/auth/register/register';

// Pages
import { Home } from './features/home/home';
import { About } from './features/about/about';

// NUEVOS COMPONENTES
import { HotelsList } from './features/hotels/hotels-list/hotels-list';
import { HotelDetail } from './features/hotels/hotel-detail/hotel-detail';

export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'home' },

  // Auth (sin shell)
  { path: 'login', component: Login },
  { path: 'register', component: Register },

  // App (con navbar)
  {
    path: '',
    component: ShellComponent,
    children: [
      { path: 'home', component: Home },

      {
        path: 'hotels',
        children: [
          { path: '', component: HotelsList },
          { path: ':hotelSlug', component: HotelDetail },
        ],
      },

      { path: 'about', component: About },
    ],
  },

  { path: '**', redirectTo: 'home' },
];
