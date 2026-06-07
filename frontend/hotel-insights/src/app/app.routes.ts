import { Routes } from '@angular/router';
import { ShellComponent } from './shared/shell/shell';

// Main app pages.
import { Home } from './features/home/home';
import { About } from './features/about/about';

// Hotel feature pages.
import { HotelsList } from './features/hotels/hotels-list/hotels-list';
import { HotelDetail } from './features/hotels/hotel-detail/hotel-detail';

export const routes: Routes = [
  // Auth is handled by an overlay modal (AuthModalComponent), not by routes.

  // App routes rendered inside the shell.
  {
    path: '',
    component: ShellComponent,
    children: [
      { path: '', component: Home, pathMatch: 'full' },
      { path: 'home', redirectTo: '', pathMatch: 'full' },

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

  { path: '**', redirectTo: '' },
];
