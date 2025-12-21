import { useEffect, useMemo, useState } from 'react'
import { supabase } from '../supabaseClient'
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

export type SentimentRow = {
  hotel: string
  year: number
  sentiment_score: number
}

type FetchState = 'idle' | 'loading' | 'loaded' | 'error'

const VIEW_NAME = import.meta.env.VITE_SENTIMENT_VIEW || 'vw_sentiment_by_year'

export function SentimentByYear() {
  const [rows, setRows] = useState<SentimentRow[]>([])
  const [state, setState] = useState<FetchState>('idle')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setState('loading')
    supabase
      .from(VIEW_NAME)
      .select('hotel, year, sentiment_score')
      .order('year', { ascending: true })
      .order('hotel', { ascending: true })
      .limit(200)
      .then(({ data, error }) => {
        if (error) {
          setError(error.message)
          setState('error')
          return
        }
        setRows(data ?? [])
        setState('loaded')
      })
  }, [])

  const chartData = useMemo(() => {
    const byYear: Record<number, Record<string, number>> = {}
    rows.forEach((row) => {
      if (!byYear[row.year]) byYear[row.year] = {}
      byYear[row.year][row.hotel] = row.sentiment_score
    })

    return Object.entries(byYear)
      .sort(([a], [b]) => Number(a) - Number(b))
      .map(([year, values]) => ({
        year: Number(year),
        ...values,
      }))
  }, [rows])

  const hotels = useMemo(() => Array.from(new Set(rows.map((r) => r.hotel))).sort(), [rows])

  return (
    <div className="card">
      <div className="grid" style={{ alignItems: 'flex-start' }}>
        <div>
          <div className="badge">Supabase view: {VIEW_NAME}</div>
          <h2 style={{ margin: '12px 0 8px' }}>Sentiment by year</h2>
          <p style={{ margin: 0, color: '#475569' }}>
            Reads aggregated sentiment scores directly from Supabase. Swap the view or columns to match your
            production schema.
          </p>
          <Status state={state} error={error} />
        </div>
      </div>

      <div style={{ marginTop: 20 }}>
        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Hotel</th>
                <th>Year</th>
                <th>Sentiment</th>
              </tr>
            </thead>
            <tbody>
              {rows.length === 0 ? (
                <tr>
                  <td colSpan={3}>No data returned from {VIEW_NAME}. Confirm the view and access policy.</td>
                </tr>
              ) : (
                rows.map((row, idx) => (
                  <tr key={`${row.hotel}-${row.year}-${idx}`}>
                    <td>{row.hotel}</td>
                    <td>{row.year}</td>
                    <td>{row.sentiment_score.toFixed(3)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div style={{ marginTop: 24, height: 320 }}>
        <ResponsiveContainer>
          <AreaChart data={chartData} margin={{ top: 10, left: 0, right: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="year" />
            <YAxis domain={[0, 1]} />
            <Tooltip formatter={(value: number) => value.toFixed(3)} />
            <Legend />
            {hotels.map((hotel) => (
              <Area
                key={hotel}
                type="monotone"
                dataKey={hotel}
                strokeWidth={2}
                stroke="#0ea5e9"
                fill="#bae6fd"
                fillOpacity={0.35}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function Status({ state, error }: { state: FetchState; error: string | null }) {
  if (state === 'error') return <p className="error">Failed to load data: {error}</p>
  if (state === 'loading') return <p className="status">Loading data from Supabase…</p>
  if (state === 'loaded') return <p className="status">Live data connected ✔</p>
  return null
}
