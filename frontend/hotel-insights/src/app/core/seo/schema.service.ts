import { DOCUMENT } from '@angular/common';
import { Inject, Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root',
})
export class SchemaService {
  constructor(@Inject(DOCUMENT) private document: Document) {}

  setSchema(id: string, schema: object): void {
    this.removeSchema(id);

    const script = this.document.createElement('script');
    script.type = 'application/ld+json';
    script.id = id;
    script.textContent = JSON.stringify(schema);

    this.document.head.appendChild(script);
  }

  removeSchema(id: string): void {
    const existingScript = this.document.getElementById(id);

    if (existingScript) {
      existingScript.remove();
    }
  }
}
