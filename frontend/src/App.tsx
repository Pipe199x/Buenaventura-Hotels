import { SentimentByYear } from './components/SentimentByYear'

export default function App() {
  return (
    <div className="app-shell">
      <header className="header">
        <div style={{ maxWidth: 1100, margin: '0 auto', padding: '0 20px' }}>
          <div className="badge">Cloudflare Pages + Supabase</div>
          <h1>Buenaventura Hotel Insights</h1>
          <p>Lightweight dashboard that reads Supabase SQL views using the public anon key.</p>
        </div>
      </header>

      <main className="main-content">
        <div className="banner">
          <span>🚀</span>
          <span>
            Deploy-ready: set <code>VITE_SUPABASE_URL</code> and <code>VITE_SUPABASE_ANON_KEY</code> in Cloudflare Pages
            environment variables, then push to <code>main</code> to auto-build.
          </span>
        </div>

        <div style={{ marginTop: 20 }}>
          <SentimentByYear />
        </div>
      </main>
    </div>
  )
}
