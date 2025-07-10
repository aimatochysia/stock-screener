export const parseTechnicalCSV = csvData => {
  const lines = csvData.trim().split('\n')
  if (lines.length < 2) return null

  const headers = lines[0].split(',').map(h => h.trim())
  const lastLine = lines[lines.length - 1]
  const values = lastLine.split(',')

  const technical = {}
  headers.forEach((header, index) => {
    const value = values[index]?.trim()
    if (header === 'date') {
      technical[header] = value
    } else if (header === 'market_stage') {
      technical[header] = value
    } else {
      technical[header] = parseFloat(value) || 0
    }
  })

  return technical
}

export const parseLevelsCSV = csvData => {
  const lines = csvData.trim().split('\n')
  if (lines.length < 2) return []

  return lines
    .slice(1)
    .map(line => {
      const price = parseFloat(line.trim())
      return {
        price: price,
        level_price: price
      }
    })
    .filter(level => !isNaN(level.price))
}

export const parseChannelCSV = csvData => {
  const lines = csvData.trim().split('\n')
  if (lines.length < 2) return null

  const headers = lines[0].split(',').map(h => h.trim())
  const lastLine = lines[lines.length - 1]
  const values = lastLine.split(',')

  const channel = {}
  headers.forEach((header, index) => {
    const value = values[index]?.trim()
    if (header.includes('date')) {
      channel[header] = value
    } else {
      channel[header] = parseFloat(value) || 0
    }
  })

  return channel
}

export const parseDailyCSV = csvData => {
  const lines = csvData.trim().split('\n')
  const headers = lines[0].split(',').map(h => h.trim().toLowerCase())

  return lines.slice(1).map(line => {
    const values = line.split(',')
    const row = {}

    headers.forEach((header, index) => {
      const value = values[index]?.trim()
      if (header === 'date') {
        row[header] = value
      } else {
        row[header] = parseFloat(value) || 0
      }
    })
    return row
  })
}
