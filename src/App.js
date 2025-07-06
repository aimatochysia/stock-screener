import React, { useState, useEffect, useMemo, useCallback } from 'react'
import StockTable from './components/StockTable'
import StockChart from './components/StockChart'
import TopContenders from './components/TopContenders'
import ProgressBar from './components/ProgressBar'
import useStockData from './hooks/useStockData'
import './App.css'

function App () {
  const [selectedStock, setSelectedStock] = useState(null)
  const [tableData, setTableData] = useState([])
  const [topStocks, setTopStocks] = useState([])
  const [page, setPage] = useState(0)
  const [filters, setFilters] = useState({})
  const [sortConfig, setSortConfig] = useState({
    key: 'rsi_14',
    direction: 'asc'
  })
  const [initialLoading, setInitialLoading] = useState(true)

  const { getStockData, getMultipleStockData, loading } = useStockData()
  const PAGE_SIZE = 20

  // Predefined list of all stock symbols
  const allSymbols = useMemo(
    () => [
      'BBNI.JK',
      'BBCA.JK',
      'BBRI.JK',
      'TLKM.JK',
      'UNVR.JK',
      'ASII.JK',
      'GOTO.JK',
      'INDF.JK',
      'ICBP.JK',
      'UNTR.JK',
      'BMRI.JK',
      'KLBF.JK'
    ],
    []
  )

  // Load initial data
  useEffect(() => {
    const loadInitialData = async () => {
      console.log('Loading initial data...')
      setInitialLoading(true)

      try {
        // Load top 5 contenders first
        const topSymbols = allSymbols.slice(0, 5)
        console.log('Loading top contenders:', topSymbols)
        const topData = await getMultipleStockData(topSymbols)
        console.log('Top contenders loaded:', topData)
        setTopStocks(
          topData.sort((a, b) => a.technical.rsi_14 - b.technical.rsi_14)
        )

        // Load first page of table data
        const initialSymbols = allSymbols.slice(0, PAGE_SIZE)
        console.log('Loading initial table data:', initialSymbols)
        const tableData = await getMultipleStockData(initialSymbols)
        console.log('Initial table data loaded:', tableData)
        setTableData(tableData)
        setPage(1)
      } catch (error) {
        console.error('Error loading initial data:', error)
      } finally {
        setInitialLoading(false)
      }
    }

    loadInitialData()
  }, [allSymbols, getMultipleStockData])

  // Load next page of data
  const loadNextPage = useCallback(async () => {
    if (loading) return

    const start = page * PAGE_SIZE
    const end = start + PAGE_SIZE
    const symbolsToLoad = allSymbols.slice(start, end)

    if (symbolsToLoad.length === 0) return

    const newData = await getMultipleStockData(symbolsToLoad)
    setTableData(prev => [...prev, ...newData])
    setPage(prev => prev + 1)
  }, [page, loading, allSymbols, getMultipleStockData])

  // Apply sorting
  const sortedData = useMemo(() => {
    const sortableItems = [...tableData]
    if (sortConfig.key) {
      sortableItems.sort((a, b) => {
        const aValue = a.technical[sortConfig.key]
        const bValue = b.technical[sortConfig.key]

        if (aValue < bValue) {
          return sortConfig.direction === 'asc' ? -1 : 1
        }
        if (aValue > bValue) {
          return sortConfig.direction === 'asc' ? 1 : -1
        }
        return 0
      })
    }
    return sortableItems
  }, [tableData, sortConfig])

  // Apply filtering
  const filteredData = useMemo(() => {
    return sortedData.filter(stock => {
      return Object.entries(filters).every(([key, value]) => {
        const stockValue = stock.technical[key]
        if (typeof stockValue === 'number') {
          return stockValue >= value.min && stockValue <= value.max
        }
        return stockValue.includes(value)
      })
    })
  }, [sortedData, filters])

  const handleRowClick = stock => {
    setSelectedStock(stock)
  }

  const handleSort = key => {
    setSortConfig({
      key,
      direction:
        sortConfig.key === key && sortConfig.direction === 'asc'
          ? 'desc'
          : 'asc'
    })
  }

  const handleFilterChange = (key, value) => {
    setFilters(prev => ({
      ...prev,
      [key]: value
    }))
  }

  if (initialLoading) {
    return (
      <div className='dashboard'>
        <div
          style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            height: '100vh',
            flexDirection: 'column'
          }}
        >
          <h2>Loading Stock Data...</h2>
          <ProgressBar
            progress={0}
            total={100}
            message='Fetching data from GitHub...'
          />
        </div>
      </div>
    )
  }

  return (
    <div className='dashboard'>
      <header className='header'>
        <h1>Stock Screener</h1>
        <div className='controls'>
          <input
            type='text'
            placeholder='Filter stocks...'
            onChange={e => handleFilterChange('symbol', e.target.value)}
          />
        </div>
      </header>

      <section className='top-contenders'>
        <TopContenders stocks={topStocks} onSelect={setSelectedStock} />
      </section>

      <main className='main-content'>
        <div className='table-container'>
          <StockTable
            data={filteredData}
            onRowClick={handleRowClick}
            onSort={handleSort}
            sortConfig={sortConfig}
            loadMore={loadNextPage}
            hasMore={page * PAGE_SIZE < allSymbols.length}
            isLoading={loading}
            selectedStock={selectedStock}
          />
        </div>

        <div className='detail-panel'>
          {selectedStock ? (
            <StockChart stock={selectedStock} />
          ) : (
            <div className='placeholder'>
              <p>Select a stock to view detailed analysis</p>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}

export default App
