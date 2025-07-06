import React from "react";

const TopContenders = ({ stocks, onSelect }) => {
  if (!stocks || stocks.length === 0) return null;

  return (
    <div className="top-contenders">
      <h3>Top Contenders</h3>
      <div className="contenders-grid">
        {stocks.map((stock) => (
          <div
            key={stock.symbol}
            className="contender-card"
            onClick={() => onSelect(stock)}
          >
            <div className="card-header">
              <span className="symbol">{stock.symbol}</span>
              <span className="price">
                ${stock.technical?.close?.toFixed(2) || "N/A"}
              </span>
            </div>
            <div className="card-body">
              <div className="metric">
                <span>RSI:</span>
                <span
                  className={
                    stock.technical?.rsi_14 < 30
                      ? "value-oversold"
                      : stock.technical?.rsi_14 > 70
                      ? "value-overbought"
                      : ""
                  }
                >
                  {stock.technical?.rsi_14?.toFixed(2) || "N/A"}
                </span>
              </div>
              <div className="metric">
                <span>Vol:</span>
                <span>
                  {stock.technical?.volume
                    ? (stock.technical.volume / 1000000).toFixed(2) + "M"
                    : "N/A"}
                </span>
              </div>
              <div className="metric">
                <span>Trend:</span>
                <span
                  className={
                    stock.technical?.market_stage === "downtrend"
                      ? "trend-down"
                      : stock.technical?.market_stage === "uptrend"
                      ? "trend-up"
                      : ""
                  }
                >
                  {stock.technical?.market_stage || "N/A"}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default TopContenders;
