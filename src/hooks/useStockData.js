import axios from 'axios'
import { useState } from 'react'
import {
  parseTechnicalCSV,
  parseLevelsCSV,
  parseChannelCSV,
  parseDailyCSV
} from '../utils/dataParser'

const stockCache = new Map()

export default function useStockData () {
  const [loading, setLoading] = useState(false)

  const TECHNICAL_BASE_URL =
    'https://raw.githubusercontent.com/aimatochysia/stock-results/refs/heads/main'

  const DAILY_BASE_URL =
    'https://raw.githubusercontent.com/aimatochysia/stock-db/refs/heads/main'

  const fetchCSV = async (url, symbol, type) => {
    try {
      console.log(`Fetching ${type} for ${symbol}: ${url}`)
      const response = await axios.get(url, {
        headers: {
          'Cache-Control': 'max-age=300',
          'If-None-Match': ''
        },
        timeout: 10000
      })
      console.log(`Successfully fetched ${type} for ${symbol}`)
      return response.data
    } catch (error) {
      if (error.response?.status === 404) {
        console.warn(`Data not found for ${symbol}_${type} at ${url}`)
      } else {
        console.error(`Error fetching ${type} for ${symbol}:`, error.message)
      }
      return null
    }
  }

  const getStockData = async symbol => {
    console.log(`Getting stock data for: ${symbol}`)

    if (stockCache.has(symbol)) {
      console.log(`Using cached data for ${symbol}`)
      return stockCache.get(symbol)
    }

    try {
      const [technical, levels, channel, daily] = await Promise.all([
        fetchCSV(
          `${TECHNICAL_BASE_URL}/${symbol}_technical.csv`,
          symbol,
          'technical'
        ),
        fetchCSV(
          `${TECHNICAL_BASE_URL}/${symbol}_levels.csv`,
          symbol,
          'levels'
        ),
        fetchCSV(
          `${TECHNICAL_BASE_URL}/${symbol}_channel.csv`,
          symbol,
          'channel'
        ),
        fetchCSV(`${DAILY_BASE_URL}/${symbol}.csv`, symbol, 'daily')
      ])

      console.log(`Raw data received for ${symbol}:`, {
        technical: technical ? 'OK' : 'MISSING',
        levels: levels ? 'OK' : 'MISSING',
        channel: channel ? 'OK' : 'MISSING',
        daily: daily ? 'OK' : 'MISSING'
      })

      const stockData = {
        symbol,
        technical: technical ? parseTechnicalCSV(technical) : null,
        levels: levels ? parseLevelsCSV(levels) : [],
        channel: channel ? parseChannelCSV(channel) : null,
        daily: daily ? parseDailyCSV(daily) : [],
        lastUpdated: Date.now()
      }

      console.log(`Parsed data for ${symbol}:`, {
        symbol: stockData.symbol,
        hasTechnical: !!stockData.technical,
        levelsCount: stockData.levels.length,
        hasChannel: !!stockData.channel,
        dailyCount: stockData.daily.length
      })

      stockCache.set(symbol, stockData)
      return stockData
    } catch (error) {
      console.error(`Error processing ${symbol}:`, error)
      return {
        symbol,
        technical: null,
        levels: [],
        channel: null,
        daily: []
      }
    }
  }

  const getMultipleStockData = async symbols => {
    console.log('Fetching multiple stocks:', symbols)
    setLoading(true)

    try {
      const requests = symbols.map(symbol => getStockData(symbol))
      const results = await Promise.all(requests)
      const filteredResults = results.filter(stock => stock.technical !== null)

      console.log(
        `Successfully loaded ${filteredResults.length} out of ${results.length} stocks`
      )
      return filteredResults
    } catch (error) {
      console.error('Error loading multiple stocks:', error)
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
