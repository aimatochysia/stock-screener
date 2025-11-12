import React from "react";

const StockTable = ({
  data,
  onRowClick,
  selectedStock = null,
}) => {
  const colorize = (value, maxAbs = 10) => {
    if (value == null || isNaN(value)) return 'transparent'

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
      color: value != null && !isNaN(value) ? '#fff' : 'inherit'
    }
  }

  return (
    <div className="stock-table-container">
      {data?.length === 0 && (
        <div className="no-data">No stocks to display</div>
      )}

      <div className="stock-table-scroll">
        <table className="stock-table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Price</th>
              <th>Volume</th>
              <th>RSI</th>
              <th>ATR%</th>
              <th>Trend</th>
              <th>MA Align</th>
              <th>% to SMA50</th>
              <th>5Δ%</th>
              <th>20Δ%</th>
              <th>50Δ%</th>
            </tr>
          </thead>
          <tbody>
            {data.map((stock) => (
              <tr
                key={stock.symbol}
                onClick={() => onRowClick(stock)}
                className={selectedStock?.symbol === stock.symbol ? "selected" : ""}
              >
                <td className="symbol-cell">{stock.symbol}</td>
                <td>{stock.close?.toFixed(0) || '-'}</td>
                <td>{stock.volume ? (stock.volume / 1000000).toFixed(1) + 'M' : '-'}</td>
                <td 
                  className={
                    stock.rsi_14 < 30 ? 'rsi-oversold' : 
                    stock.rsi_14 > 70 ? 'rsi-overbought' : ''
                  }
                >
                  {stock.rsi_14?.toFixed(1) || '-'}
                </td>
                <td>{stock.atr_pct?.toFixed(2) || '-'}</td>
                <td 
                  className={
                    stock.market_stage === 'uptrend' ? 'trend-up' :
                    stock.market_stage === 'downtrend' ? 'trend-down' : ''
                  }
                >
                  {stock.market_stage || '-'}
                </td>
                <td>{stock.ma_alignment || '-'}</td>
                <td style={getCellStyle(stock.price_vs_sma_50_pct, stock.atr_pct)}>
                  {stock.price_vs_sma_50_pct?.toFixed(1) || '-'}%
                </td>
                <td style={getCellStyle(stock.sma_5_diff_pct, stock.atr_pct)}>
                  {stock.sma_5_diff_pct?.toFixed(1) || '-'}%
                </td>
                <td style={getCellStyle(stock.sma_20_diff_pct, stock.atr_pct)}>
                  {stock.sma_20_diff_pct?.toFixed(1) || '-'}%
                </td>
                <td style={getCellStyle(stock.sma_50_diff_pct, stock.atr_pct)}>
                  {stock.sma_50_diff_pct?.toFixed(1) || '-'}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default StockTable;
