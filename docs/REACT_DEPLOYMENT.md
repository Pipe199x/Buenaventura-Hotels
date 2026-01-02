# Guía de Despliegue - Aplicación React a Cloudflare

Guía completa para desplegar la aplicación React de análisis de hoteles a Cloudflare Pages.

## 📋 Requisitos Previos

1. **Cuenta de Cloudflare** (gratuita funciona)
2. **Repositorio en GitHub/GitLab**
3. **Proyecto Supabase** con datos cargados
4. **Node.js 18+** instalado localmente

## 🚀 Paso 1: Preparar el Código

1. **Asegúrate de que tu código esté en Git:**
   ```bash
   cd web-react
   git add .
   git commit -m "Add React application"
   git push origin main
   ```

2. **Verifica que el build funciona localmente:**
   ```bash
   npm install
   npm run build
   ```
   
   Si funciona, deberías ver una carpeta `dist/` creada.

## 🌐 Paso 2: Conectar GitHub a Cloudflare Pages

### Método A: Desde Cloudflare Dashboard (Recomendado)

1. **Ve a Cloudflare Dashboard:**
   - Inicia sesión en [dash.cloudflare.com](https://dash.cloudflare.com)
   - Navega a **Pages** en el menú lateral

2. **Crear nuevo proyecto:**
   - Click en **Create a project**
   - Selecciona **Connect to Git**
   - Autoriza Cloudflare a acceder a tu GitHub/GitLab

3. **Seleccionar repositorio:**
   - Elige tu repositorio
   - Selecciona la rama (generalmente `main`)

4. **Configurar build settings:**
   ```
   Framework preset: Vite
   Build command: npm run build
   Build output directory: dist
   Root directory: web-react
   ```

5. **Variables de entorno:**
   - Click en **Environment variables**
   - Agrega:
     - `VITE_SUPABASE_URL`: `https://tu-proyecto.supabase.co`
     - `VITE_SUPABASE_ANON_KEY`: Tu anon key de Supabase
   
   **Importante:** Para producción, marca estas variables como "Production" y "Preview"

6. **Desplegar:**
   - Click en **Save and Deploy**
   - Cloudflare construirá y desplegará tu aplicación automáticamente

### Método B: Desde Wrangler CLI

1. **Instalar Wrangler:**
   ```bash
   npm install -g wrangler
   ```

2. **Login:**
   ```bash
   wrangler login
   ```

3. **Crear proyecto de Pages:**
   ```bash
   cd web-react
   wrangler pages project create hotel-insights
   ```

4. **Desplegar:**
   ```bash
   npm run build
   wrangler pages deploy dist --project-name=hotel-insights
   ```

5. **Configurar secrets:**
   ```bash
   wrangler pages secret put VITE_SUPABASE_URL --project-name=hotel-insights
   wrangler pages secret put VITE_SUPABASE_ANON_KEY --project-name=hotel-insights
   ```

## 🔧 Paso 3: Configurar Dominio Personalizado (Opcional)

1. **En Cloudflare Pages:**
   - Ve a tu proyecto → **Custom domains**
   - Click en **Set up a custom domain**
   - Ingresa tu dominio (ej: `analytics.tudominio.com`)

2. **Configurar DNS:**
   - Cloudflare te dará un registro CNAME
   - Agrega el registro en tu DNS
   - SSL se configurará automáticamente

## ✅ Paso 4: Verificar Despliegue

1. **Visita tu URL:**
   - Cloudflare Pages URL: `https://tu-proyecto.pages.dev`
   - O tu dominio personalizado

2. **Verifica funcionalidades:**
   - ✅ La página carga correctamente
   - ✅ Los datos se cargan desde Supabase
   - ✅ Las gráficas se muestran
   - ✅ La navegación funciona

## 🔄 Actualizaciones Automáticas

Una vez configurado, cada vez que hagas `git push` a tu rama principal:

1. Cloudflare detectará el cambio
2. Ejecutará el build automáticamente
3. Desplegará la nueva versión
4. Tu sitio se actualizará en ~2-3 minutos

## 🐛 Solución de Problemas

### Build falla

**Error:** `npm: command not found`
- **Solución:** Asegúrate de que el build command sea `npm run build` y no solo `npm build`

**Error:** `Module not found`
- **Solución:** Verifica que todas las dependencias estén en `package.json`

**Error:** Variables de entorno no encontradas
- **Solución:** Verifica que las variables estén configuradas en Cloudflare Pages → Settings → Environment Variables

### La aplicación carga pero no hay datos

**Problema:** Error de CORS o conexión a Supabase
- **Solución:** 
  1. Verifica que las variables de entorno estén correctas
  2. Revisa que RLS en Supabase permita acceso público
  3. Verifica la consola del navegador para errores específicos

### Gráficas no se muestran

**Problema:** Recharts no carga
- **Solución:** 
  1. Verifica que Recharts esté en `package.json`
  2. Revisa la consola del navegador
  3. Asegúrate de que los datos se estén cargando correctamente

## 📊 Monitoreo

1. **Analytics en Cloudflare:**
   - Ve a tu proyecto → Analytics
   - Verás tráfico, visitas, y métricas de rendimiento

2. **Logs de build:**
   - Ve a Deployments → Click en un deployment
   - Verás logs completos del proceso de build

3. **Supabase Dashboard:**
   - Monitorea queries y uso de la base de datos
   - Revisa logs de errores si los hay

## 🔒 Seguridad

1. **Variables de entorno:**
   - Nunca commitees `.env` a Git
   - Usa siempre variables de entorno en Cloudflare
   - No uses el service role key, solo el anon key

2. **Row Level Security:**
   - Configura RLS en Supabase para proteger datos sensibles
   - Solo permite lectura pública de datos necesarios

## 📝 Checklist de Despliegue

- [ ] Código en GitHub/GitLab
- [ ] Build funciona localmente (`npm run build`)
- [ ] Variables de entorno configuradas en Cloudflare
- [ ] Proyecto conectado a repositorio
- [ ] Build settings correctos
- [ ] Despliegue exitoso
- [ ] Aplicación funciona en producción
- [ ] Dominio personalizado configurado (opcional)
- [ ] SSL activo

## 🎉 ¡Listo!

Tu aplicación React está desplegada en Cloudflare Pages. Cada push a tu repositorio actualizará automáticamente el sitio.

Para más información, consulta:
- [Cloudflare Pages Docs](https://developers.cloudflare.com/pages/)
- [Vite Deployment Guide](https://vitejs.dev/guide/static-deploy.html)

