import { ComponentFixture, TestBed } from '@angular/core/testing';
import { vi } from 'vitest';
import { provideRouter } from '@angular/router';

import { Research } from './research';
import { RESEARCH, SITE_ORIGIN } from '../../core/seo/hotels.metadata';

// jsdom has no real scrollTo, and innerWidth is not a settable accessor there.
window.scrollTo = vi.fn();

// jsdom exposes innerWidth as a plain property, so redefine it instead of spying.
function setViewportWidth(width: number): void {
  Object.defineProperty(window, 'innerWidth', { value: width, configurable: true });
}

describe('Research', () => {
  let component: Research;
  let fixture: ComponentFixture<Research>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Research],
      providers: [provideRouter([])]
    })
    .compileComponents();

    fixture = TestBed.createComponent(Research);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  afterEach(() => {
    document.getElementById('schema-research-article')?.remove();
    document.getElementById('schema-research-breadcrumb')?.remove();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  // The whole point of this page is the structured data linking the site to the
  // published thesis. If that JSON-LD stops being emitted, the page is dead weight.
  it('emits ScholarlyArticle JSON-LD pointing at the RIDUM record', () => {
    const el = document.getElementById('schema-research-article');
    expect(el).toBeTruthy();

    const schema = JSON.parse(el!.textContent ?? '{}');
    expect(schema['@type']).toBe('ScholarlyArticle');
    expect(schema['@id']).toBe(`${SITE_ORIGIN}/investigacion#thesis`);
    expect(schema.sameAs).toContain(RESEARCH.handleUrl);
    expect(schema.author['@id']).toBe(`${SITE_ORIGIN}/#author`);
    expect(schema.encoding.contentUrl).toBe(RESEARCH.pdfUrl);
  });

  // A repository URL carrying a session token must never reach the published page.
  it('never exposes a tokenised repository URL', () => {
    const values = Object.values(RESEARCH).join(' ');
    expect(values).not.toContain('authentication-token');
  });

  // toggleViewer has the only branch on this page: embed inline when the browser
  // can render a PDF in a frame, hand off to a new tab when it cannot.
  it('opens and closes the inline viewer on a wide viewport', () => {
    setViewportWidth(1280);

    expect(component.viewerOpen).toBe(false);
    component.toggleViewer();
    expect(component.viewerOpen).toBe(true);
    component.toggleViewer();
    expect(component.viewerOpen).toBe(false);
  });

  it('falls back to a new tab on a narrow viewport', () => {
    setViewportWidth(500);
    const open = vi.spyOn(window, 'open').mockReturnValue(null);

    component.toggleViewer();

    expect(component.viewerOpen).toBe(false);
    expect(open).toHaveBeenCalledWith(RESEARCH.localPdfPath, '_blank', 'noopener');
  });
});
