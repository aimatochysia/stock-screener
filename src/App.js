// src/App.js
import React, { useEffect, useState, useMemo } from 'react'
import useStockData from './hooks/useStockData'
import './App.css'

function App() {
  const { getAllLatestTechnical, loading } = useStockData()
  const [technicalData, setTechnicalData] = useState([])
  const [search, setSearch] = useState('')

  useEffect(() => {
    const loadData = async () => {
      const raw = await getAllLatestTechnical()
      const parsed = Object.entries(raw).map(([symbolKey, data]) => ({
        symbol: symbolKey.replace('.json', ''),
        ...data
      }))
      setTechnicalData(parsed)
    }

    loadData()
  }, [])

  const filteredData = useMemo(() => {
    return technicalData.filter(stock =>
      stock.symbol.toLowerCase().includes(search.toLowerCase())
    )
  }, [technicalData, search])

  const colorize = (value, maxAbs = 10) => {
    if (value == null || isNaN(value)) return 'unset'

    const ratio = Math.max(-1, Math.min(1, value / maxAbs))
    const green = ratio > 0 ? Math.floor(80 + 100 * ratio) : 80
    const red = ratio < 0 ? Math.floor(80 - 100 * ratio) : 80
    const color = `rgb(${red}, ${green}, 80)`
    return color
  }

  const getCellStyle = (value, atrPct) => {
    const scale = atrPct || 5
    return {
      backgroundColor: colorize(value, scale),
      color: '#fff'
    }
  }

  return (
    <div className="dark-container">
      <h1>📊 Technical Screener</h1>

      <input
        className="search-input"
        placeholder="Search symbol..."
        value={search}
        onChange={e => setSearch(e.target.value)}
      />

      {loading && <p>Loading data...</p>}

      <div className="scroll-container">
        <table className="screener-table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Close</th>
              <th>Volume</th>
              <th>ATR%</th>
              <th>RSI</th>
              <th>OB</th>
              <th>OS</th>
              <th>% to SMA50</th>
              <th>MA Align</th>
              <th>Stage</th>
              <th>SMA5</th>
              <th>5Δ%</th>
              <th>SMA10</th>
              <th>10Δ%</th>
              <th>SMA20</th>
              <th>20Δ%</th>
              <th>SMA50</th>
              <th>50Δ%</th>
              <th>SMA100</th>
              <th>100Δ%</th>
              <th>SMA200</th>
              <th>200Δ%</th>
            </tr>
          </thead>
          <tbody>
            {filteredData.map(stock => (
              <tr key={stock.symbol}>
                <td>{stock.symbol}</td>
                <td>{stock.close ?? '-'}</td>
                <td>{stock.volume}</td>
                <td>{stock.atr_pct?.toFixed(2) ?? '-'}</td>
                <td style={getCellStyle(stock.rsi_14 - 50, stock.atr_pct)}>
                  {stock.rsi_14?.toFixed(2) ?? '-'}
                </td>
                <td>{stock.rsi_overbought}</td>
                <td>{stock.rsi_oversold}</td>
                <td style={getCellStyle(stock.price_vs_sma_50_pct, stock.atr_pct)}>
                  {stock.price_vs_sma_50_pct?.toFixed(2) ?? '-'}
                </td>
                <td>{stock.ma_alignment || '-'}</td>
                <td>{stock.market_stage || '-'}</td>

                {/* SMA5 */}
                <td>{stock.sma_5 ?? '-'}</td>
                <td style={getCellStyle(stock.sma_5_diff_pct, stock.atr_pct)}>
                  {stock.sma_5_diff_pct?.toFixed(2) ?? '-'}%
                </td>

                {/* SMA10 */}
                <td>{stock.sma_10 ?? '-'}</td>
                <td style={getCellStyle(stock.sma_10_diff_pct, stock.atr_pct)}>
                  {stock.sma_10_diff_pct?.toFixed(2) ?? '-'}%
                </td>

                {/* SMA20 */}
                <td>{stock.sma_20 ?? '-'}</td>
                <td style={getCellStyle(stock.sma_20_diff_pct, stock.atr_pct)}>
                  {stock.sma_20_diff_pct?.toFixed(2) ?? '-'}%
                </td>

                {/* SMA50 */}
                <td>{stock.sma_50 ?? '-'}</td>
                <td style={getCellStyle(stock.sma_50_diff_pct, stock.atr_pct)}>
                  {stock.sma_50_diff_pct?.toFixed(2) ?? '-'}%
                </td>

                {/* SMA100 */}
                <td>{stock.sma_100 ?? '-'}</td>
                <td style={getCellStyle(stock.sma_100_diff_pct, stock.atr_pct)}>
                  {stock.sma_100_diff_pct?.toFixed(2) ?? '-'}%
                </td>

                {/* SMA200 */}
                <td>{stock.sma_200 ?? '-'}</td>
                <td style={getCellStyle(stock.sma_200_diff_pct, stock.atr_pct)}>
                  {stock.sma_200_diff_pct?.toFixed(2) ?? '-'}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default App
