import { Component, OnInit } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { AuthService } from './core/auth.service';
import { TranslateService } from '@ngx-translate/core';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet],
  template: `<router-outlet></router-outlet>`,
})
export class AppComponent implements OnInit {
  constructor(private auth: AuthService, private translate: TranslateService) {
    const saved = localStorage.getItem('lang') || 'es';
    this.translate.setDefaultLang('es');
    this.translate.use(saved);
  }

  ngOnInit() {
    this.auth.init();
  }
}
