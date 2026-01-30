---
applyTo: '**'
---
Provide project context and coding guidelines that AI should follow when generating code, answering questions, or reviewing changes.

🧭 instructions.md

AI / Copilot Project Instructions – Hotel Insights

1. Contexto del proyecto

Este repositorio corresponde al proyecto Hotel Insights, una aplicación web construida con:

Frontend: Angular (standalone components, Angular 17+)

Backend/Data: Supabase (PostgreSQL + Auth)

Auth: Email/Password + Google OAuth (Supabase)

Objetivo: Visualizar análisis de opiniones hoteleras (KPIs, gráficos, vistas agregadas)

El proyecto se desarrolla de forma iterativa y ordenada, con foco académico y profesional.

2. Principios generales (OBLIGATORIOS)

Cualquier código generado DEBE cumplir:

✅ Clean Code

Nombres claros, en inglés

Funciones cortas y con una sola responsabilidad

Nada de “magic values” → usar constantes

✅ SOLID

S – Single Responsibility:
Un servicio, componente o función hace una sola cosa.

O – Open/Closed:
El código debe poder extenderse sin modificarse.

L – Liskov Substitution:
Evitar herencias frágiles; preferir composición.

I – Interface Segregation:
No forzar dependencias innecesarias.

D – Dependency Inversion:
Los componentes dependen de servicios, no de implementaciones concretas.

3. Arquitectura Frontend (Angular)
📁 Estructura base (respetar siempre)
src/app/
├── core/               # Servicios singleton (auth, supabase, guards)
│   ├── auth.service.ts
│   ├── supabase/
│   │   └── supabase.client.ts
│   └── guards/
│       └── auth.guard.ts
│
├── features/           # Páginas / casos de uso
│   ├── auth/
│   │   ├── login/
│   │   └── register/
│   ├── home/
│   ├── hotels/
│   └── about/
│
├── shared/             # Componentes reutilizables
│   ├── navbar/
│   └── ui/
│
├── app.component.ts
├── app.routes.ts
└── app.config.ts


⚠️ NO mezclar responsabilidades

features → pantallas

core → lógica global

shared → UI reutilizable

4. Standalone Components (OBLIGATORIO)

❌ No usar NgModule

✅ Usar standalone: true

Importar explícitamente:

CommonModule

FormsModule / ReactiveFormsModule

RouterLink, RouterOutlet, etc.

Ejemplo correcto:

@Component({
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink],
})

5. Servicios (Auth, Data, etc.)
AuthService

Fuente única de verdad de la sesión

Exponer estado como Observable

No usar lógica de UI dentro del servicio

session$: Observable<Session | null>;


Los componentes se suscriben, no preguntan directamente.

6. Routing y Guards

Las rutas se definen solo en app.routes.ts

La protección de rutas se hace con functional guards

Nunca poner lógica de auth dentro de componentes de páginas

Ejemplo:

canActivate: [authGuard]

7. UI / UX

Login y Register:

Loading states

Mensajes claros

UX limpia (sin lógica innecesaria)

Navbar:

Reacciona al estado de sesión

No contiene lógica de negocio

8. Commits (MUY IMPORTANTE)
🧾 Reglas de commits

Un commit = una responsabilidad

Mensajes claros y en inglés

Prefijos recomendados:

feat:     nueva funcionalidad
fix:      corrección de bug
style:    CSS / UI
refactor: mejora sin cambiar comportamiento
config:   configuración del proyecto
chore:    tareas técnicas


Ejemplos correctos:

feat(auth): implement email login
style(login): improve login layout
config(app): setup router and http client

9. Qué NO hacer

🚫 NO:

Mezclar lógica de backend en frontend

Usar any sin razón

Escribir todo en un solo archivo

Saltarse pasos arquitectónicos

Reescribir archivos completos sin justificación

10. Forma de trabajar esperada (para Copilot)

Cuando generes código:

Pregunta si hay duda

Genera código mínimo y correcto

Explica brevemente qué hace

No avances a la siguiente etapa sin confirmación

11. Estado actual del proyecto (para referencia)

✅ Auth email/password funcionando

✅ Google OAuth funcionando (Supabase)

✅ Login/Register estilizados

⏸️ i18n pausado

🔜 Navbar + Guards + Home KPIs