import axios from 'axios'
import { useState } from 'react'

const stockCache = new Map()

export default function useStockData () {
  const [loading, setLoading] = useState(false)

  const TECHNICAL_BASE_URL =
    'https://raw.githubusercontent.com/aimatochysia/stock-results/refs/heads/main'
  const DAILY_BASE_URL =
    'https://raw.githubusercontent.com/aimatochysia/stock-db/refs/heads/main'

  const isLocalhost = () =>
    typeof window !== 'undefined' &&
    (window.location.hostname === 'localhost' ||
      window.location.hostname === '127.0.0.1')

  const fetchJSON = async (url, symbol, type) => {
    try {
      let fetchUrl = url
      if (isLocalhost()) {
        const corsProxy = 'https://corsproxy.io/?'
        fetchUrl = corsProxy + encodeURIComponent(url)
      }

      console.log(`Fetching ${type} for ${symbol}: ${fetchUrl}`)
      const response = await axios.get(fetchUrl, {
        headers: {
          'Cache-Control': 'max-age=300',
          'If-None-Match': ''
        },
        timeout: 10000
      })

      if (!response.data) {
        console.warn(`Empty response for ${type} of ${symbol}`)
        return null
      }

      return response.data
    } catch (error) {
      if (error.response?.status === 404) {
        console.warn(`404 for ${symbol} - ${type}: ${url}`)
      } else {
        console.error(`Fetch error for ${symbol} (${type}):`, error.message)
      }
      return null
    }
  }

  const getLatestTechnicalFile = async () => {
    const today = new Date()

    for (let i = 0; i < 100; i++) {
      const tryDate = new Date(today)
      tryDate.setDate(today.getDate() - i)

      const yyyy = tryDate.getFullYear()
      const mm = String(tryDate.getMonth() + 1).padStart(2, '0')
      const dd = String(tryDate.getDate()).padStart(2, '0')
      const fileName = `${yyyy}-${mm}-${dd}_technical_indicators.json`

      const url = `${TECHNICAL_BASE_URL}/${fileName}`

      try {
        const res = await axios.get(url, { timeout: 5000 })
        console.log(`Found latest technical file: ${fileName}`)
        return res.data
      } catch (err) {
        if (err.response?.status === 404) continue
        throw new Error(`Error checking ${fileName}: ${err.message}`)
      }
    }

    throw new Error(
      'Could not find a technical indicators file in the past 100 days.'
    )
  }

  const getStockData = async symbol => {
    if (stockCache.has(symbol)) {
      const cached = stockCache.get(symbol)
      if (cached?.technical) {
        return cached
      }
      stockCache.delete(symbol)
    }

    try {
      const [dailyJson, levelsChannelJson, technicalsJson] = await Promise.all([
        fetchJSON(`${DAILY_BASE_URL}/${symbol}.json`, symbol, 'daily'),
        fetchJSON(
          `${TECHNICAL_BASE_URL}/l_and_c/${symbol}.json`,
          symbol,
          'levels_channel'
        ),
        getLatestTechnicalFile()
      ])

      const daily = dailyJson?.data || []
      const levels =
        levelsChannelJson?.latest_levels?.map(p => ({
          price: p,
          level_price: p
        })) || []
      const channel = levelsChannelJson?.channel || null
      const technical = technicalsJson?.[`${symbol}.json`] || null

      if (!technical) {
        console.warn(`No technical data found for ${symbol}.`)
        return { symbol, technical: null, levels: [], channel: null, daily: [] }
      }

      const stockData = {
        symbol,
        technical,
        levels,
        channel,
        daily,
        lastUpdated: Date.now()
      }

      stockCache.set(symbol, stockData)
      return stockData
    } catch (error) {
      console.error(`Error processing ${symbol}:`, error.message)
      return { symbol, technical: null, levels: [], channel: null, daily: [] }
    }
  }

  const getMultipleStockData = async symbols => {
    setLoading(true)
    try {
      const results = await Promise.all(symbols.map(getStockData))
      return results.filter(s => s.technical !== null)
    } catch (error) {
      console.error('Batch fetch error:', error)
      return []
    } finally {
      setLoading(false)
    }
  }

  return {
    getStockData,
    getMultipleStockData,
    loading
  }
}
