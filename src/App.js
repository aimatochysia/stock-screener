// src/App.js
import React, { useEffect, useState, useMemo, useCallback } from 'react'
import useStockData from './hooks/useStockData'
import StockTable from './components/StockTable'
import StockTreeMap from './components/StockTreeMap'
import FilterPanel from './components/FilterPanel'
import MarketOverview from './components/MarketOverview'
import './App.css'

function App() {
  const { getAllLatestTechnical, loading } = useStockData()
  const [technicalData, setTechnicalData] = useState([])
  const [search, setSearch] = useState('')
  const [selectedStock, setSelectedStock] = useState(null)
  const [viewMode, setViewMode] = useState('table') // 'table' or 'map'
  const [filters, setFilters] = useState({
    rsiMin: '',
    rsiMax: '',
    volumeMin: '',
    priceMin: '',
    priceMax: '',
    marketStage: '',
    maAlignment: '',
    sma50Min: '',
    sma50Max: '',
    atrMin: '',
    atrMax: '',
  })

  // Cache the data to minimize API calls
  useEffect(() => {
    const loadData = async () => {
      // Check if data is already cached in sessionStorage
      const cachedData = sessionStorage.getItem('stockData')
      const cachedTime = sessionStorage.getItem('stockDataTime')
      const now = Date.now()
      
      // Use cache if it's less than 5 minutes old
      if (cachedData && cachedTime && (now - parseInt(cachedTime)) < 5 * 60 * 1000) {
        setTechnicalData(JSON.parse(cachedData))
        return
      }

      const raw = await getAllLatestTechnical()
      
      // If API fails or returns empty, use mock data for demo
      if (!raw || Object.keys(raw).length === 0) {
        const mockData = generateMockData()
        setTechnicalData(mockData)
        // Cache mock data too
        sessionStorage.setItem('stockData', JSON.stringify(mockData))
        sessionStorage.setItem('stockDataTime', now.toString())
        return
      }
      
      const parsed = Object.entries(raw).map(([symbolKey, data]) => ({
        symbol: symbolKey.replace('.json', ''),
        ...data
      }))
      setTechnicalData(parsed)
      
      // Cache the data
      sessionStorage.setItem('stockData', JSON.stringify(parsed))
      sessionStorage.setItem('stockDataTime', now.toString())
    }

    loadData()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Generate mock data for demo purposes
  const generateMockData = () => {
    const symbols = [
      'BBCA.JK', 'BBRI.JK', 'BMRI.JK', 'TLKM.JK', 'ASII.JK', 
      'UNVR.JK', 'ICBP.JK', 'INDF.JK', 'KLBF.JK', 'GGRM.JK',
      'SMGR.JK', 'PGAS.JK', 'PTBA.JK', 'ADRO.JK', 'INCO.JK',
      'ANTM.JK', 'WIKA.JK', 'WSKT.JK', 'BSDE.JK', 'PWON.JK',
      'SCMA.JK', 'BBTN.JK', 'HMSP.JK', 'EXCL.JK', 'JSMR.JK'
    ]
    
    const stages = ['uptrend', 'downtrend', 'sideways', 'neutral']
    const alignments = ['bullish', 'bearish', 'mixed']
    
    return symbols.map(symbol => ({
      symbol,
      close: 1000 + Math.random() * 9000,
      volume: (Math.random() * 50 + 10) * 1000000,
      rsi_14: 30 + Math.random() * 40,
      atr_pct: Math.random() * 5 + 1,
      market_stage: stages[Math.floor(Math.random() * stages.length)],
      ma_alignment: alignments[Math.floor(Math.random() * alignments.length)],
      price_vs_sma_50_pct: (Math.random() - 0.5) * 20,
      sma_5_diff_pct: (Math.random() - 0.5) * 10,
      sma_20_diff_pct: (Math.random() - 0.5) * 15,
      sma_50_diff_pct: (Math.random() - 0.5) * 20,
      sma_5: 1000 + Math.random() * 9000,
      sma_10: 1000 + Math.random() * 9000,
      sma_20: 1000 + Math.random() * 9000,
      sma_50: 1000 + Math.random() * 9000,
      sma_100: 1000 + Math.random() * 9000,
      sma_200: 1000 + Math.random() * 9000,
    }))
  }

  const handleFilterChange = useCallback((key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }))
  }, [])

  const handleResetFilters = useCallback(() => {
    setFilters({
      rsiMin: '',
      rsiMax: '',
      volumeMin: '',
      priceMin: '',
      priceMax: '',
      marketStage: '',
      maAlignment: '',
      sma50Min: '',
      sma50Max: '',
      atrMin: '',
      atrMax: '',
    })
  }, [])

  const filteredData = useMemo(() => {
    let result = technicalData

    // Search filter
    if (search) {
      result = result.filter(stock =>
        stock.symbol.toLowerCase().includes(search.toLowerCase())
      )
    }

    // RSI filter
    if (filters.rsiMin) {
      result = result.filter(stock => (stock.rsi_14 || 0) >= parseFloat(filters.rsiMin))
    }
    if (filters.rsiMax) {
      result = result.filter(stock => (stock.rsi_14 || 0) <= parseFloat(filters.rsiMax))
    }

    // Volume filter
    if (filters.volumeMin) {
      result = result.filter(stock => (stock.volume || 0) >= parseFloat(filters.volumeMin) * 1000000)
    }

    // Price filter
    if (filters.priceMin) {
      result = result.filter(stock => (stock.close || 0) >= parseFloat(filters.priceMin))
    }
    if (filters.priceMax) {
      result = result.filter(stock => (stock.close || 0) <= parseFloat(filters.priceMax))
    }

    // Market stage filter
    if (filters.marketStage) {
      result = result.filter(stock => stock.market_stage === filters.marketStage)
    }

    // MA Alignment filter
    if (filters.maAlignment) {
      result = result.filter(stock => stock.ma_alignment === filters.maAlignment)
    }

    // SMA50 filter
    if (filters.sma50Min) {
      result = result.filter(stock => (stock.price_vs_sma_50_pct || 0) >= parseFloat(filters.sma50Min))
    }
    if (filters.sma50Max) {
      result = result.filter(stock => (stock.price_vs_sma_50_pct || 0) <= parseFloat(filters.sma50Max))
    }

    // ATR filter
    if (filters.atrMin) {
      result = result.filter(stock => (stock.atr_pct || 0) >= parseFloat(filters.atrMin))
    }
    if (filters.atrMax) {
      result = result.filter(stock => (stock.atr_pct || 0) <= parseFloat(filters.atrMax))
    }

    return result
  }, [technicalData, search, filters])

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>📊 JKSE Stock Screener</h1>
        <div className="view-toggle">
          <button 
            className={viewMode === 'table' ? 'active' : ''}
            onClick={() => setViewMode('table')}
          >
            📋 Table View
          </button>
          <button 
            className={viewMode === 'map' ? 'active' : ''}
            onClick={() => setViewMode('map')}
          >
            🗺️ Market Map
          </button>
        </div>
      </header>

      <div className="app-content">
        <aside className="sidebar">
          <div className="search-box">
            <input
              className="search-input"
              placeholder="🔍 Search symbol..."
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>
          
          <FilterPanel
            filters={filters}
            onFilterChange={handleFilterChange}
            onReset={handleResetFilters}
          />

          {selectedStock && (
            <div className="stock-details">
              <h3>Stock Details</h3>
              <div className="detail-card">
                <h4>{selectedStock.symbol}</h4>
                <div className="detail-grid">
                  <div className="detail-item">
                    <span>Price:</span>
                    <span>${selectedStock.close?.toFixed(2) || 'N/A'}</span>
                  </div>
                  <div className="detail-item">
                    <span>Volume:</span>
                    <span>{(selectedStock.volume / 1000000)?.toFixed(2) || 'N/A'}M</span>
                  </div>
                  <div className="detail-item">
                    <span>RSI:</span>
                    <span className={
                      selectedStock.rsi_14 < 30 ? 'rsi-oversold' :
                      selectedStock.rsi_14 > 70 ? 'rsi-overbought' : ''
                    }>
                      {selectedStock.rsi_14?.toFixed(2) || 'N/A'}
                    </span>
                  </div>
                  <div className="detail-item">
                    <span>ATR%:</span>
                    <span>{selectedStock.atr_pct?.toFixed(2) || 'N/A'}%</span>
                  </div>
                  <div className="detail-item">
                    <span>Trend:</span>
                    <span className={
                      selectedStock.market_stage === 'uptrend' ? 'trend-up' :
                      selectedStock.market_stage === 'downtrend' ? 'trend-down' : ''
                    }>
                      {selectedStock.market_stage || 'N/A'}
                    </span>
                  </div>
                  <div className="detail-item">
                    <span>MA Align:</span>
                    <span>{selectedStock.ma_alignment || 'N/A'}</span>
                  </div>
                  <div className="detail-item">
                    <span>% to SMA50:</span>
                    <span>{selectedStock.price_vs_sma_50_pct?.toFixed(2) || 'N/A'}%</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </aside>

        <main className="main-content">
          <MarketOverview stocks={filteredData} />
          
          {loading && <div className="loading-indicator">Loading data...</div>}
          
          {!loading && (
            <div className="results-info">
              Showing {filteredData.length} of {technicalData.length} stocks
            </div>
          )}

          {viewMode === 'table' ? (
            <StockTable
              data={filteredData}
              onRowClick={setSelectedStock}
              selectedStock={selectedStock}
            />
          ) : (
            <StockTreeMap stocks={filteredData} />
          )}
        </main>
      </div>
    </div>
  )
}

export default App
