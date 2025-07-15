// src/hooks/useStockData.js
import axios from 'axios'
import { useState } from 'react'

export default function useStockData() {
  const [loading, setLoading] = useState(false)

  const getAllLatestTechnical = async () => {
    setLoading(true)
    try {
      const res = await axios.get(
        'https://stock-results.vercel.app/api/technical/latest',
        { timeout: 10000 }
      )

      if (!res.data) return {}
      return res.data // Format: { "BBCA.JK.json": {...}, ... }
    } catch (error) {
      console.error('Failed to fetch technicals:', error.message)
      return {}
    } finally {
      setLoading(false)
    }
  }

  return {
    getAllLatestTechnical,
    loading
  }
}
