import React, { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

const StockChart = ({ stock }) => {
  const chartData = useMemo(() => {
    if (!stock || !stock.daily || stock.daily.length === 0) return [];

    // Use actual daily data from stock-db
    return stock.daily.slice(-60).map((dayData, index) => ({
      date: dayData.date,
      close: parseFloat(dayData.close),
      volume: parseInt(dayData.volume),
      high: parseFloat(dayData.high),
      low: parseFloat(dayData.low),
      open: parseFloat(dayData.open),
    }));
  }, [stock]);

  if (!stock) return null;

  const currentPrice = stock.technical?.close || 0;
  const channel = stock.channel;
  const latestDaily = stock.daily?.[stock.daily.length - 1];
  const prevDaily = stock.daily?.[stock.daily.length - 2];
  const priceChange =
    latestDaily && prevDaily
      ? ((latestDaily.close - prevDaily.close) / prevDaily.close) * 100
      : 0;

  return (
    <div className="stock-chart">
      <div className="chart-header">
        <h2>{stock.symbol}</h2>
        <div className="price-info">
          <span className="price">${currentPrice.toFixed(2)}</span>
          <span
            className={`change ${priceChange >= 0 ? "positive" : "negative"}`}
          >
            {priceChange >= 0 ? "+" : ""}
            {priceChange.toFixed(2)}%
          </span>
        </div>
      </div>

      <div className="chart-container">
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f5f5f5" />
            <XAxis dataKey="date" tick={{ fontSize: 12 }} tickCount={10} />
            <YAxis
              domain={["auto", "auto"]}
              tick={{ fontSize: 12 }}
              tickFormatter={(value) => value.toFixed(2)}
            />
            <Tooltip
              formatter={(value) => [value.toFixed(2), "Price"]}
              labelFormatter={(label) => `Date: ${label}`}
            />
            <Legend />

            {/* Price Line */}
            <Line
              type="monotone"
              dataKey="close"
              stroke="#8884d8"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 6 }}
              name="Price"
            />

            {/* Support/Resistance Levels from JSON */}
            {stock.levels.map((level, index) => (
              <ReferenceLine
                key={`level-${index}`}
                y={level.price}
                stroke={level.price > currentPrice ? "#ff7300" : "#00cc66"}
                strokeDasharray="3 3"
                label={{
                  value: level.price.toFixed(2),
                  position: level.price > currentPrice ? "top" : "bottom",
                  fill: level.price > currentPrice ? "#ff7300" : "#00cc66",
                }}
              />
            ))}

            {/* Price Channel from _channel.csv */}
            {channel && (
              <>
                <ReferenceLine
                  y={parseFloat(channel.end_upper)}
                  stroke="#0088fe"
                  strokeDasharray="3 3"
                  label={{
                    value: `Upper: ${parseFloat(channel.end_upper).toFixed(2)}`,
                    position: "top",
                  }}
                />
                <ReferenceLine
                  y={parseFloat(channel.end_lower)}
                  stroke="#0088fe"
                  strokeDasharray="3 3"
                  label={{
                    value: `Lower: ${parseFloat(channel.end_lower).toFixed(2)}`,
                    position: "bottom",
                  }}
                />
              </>
            )}

            {/* Current Price Line */}
            <ReferenceLine
              y={currentPrice}
              stroke="#000"
              strokeDasharray="3 3"
              label={{
                value: `Current: ${currentPrice.toFixed(2)}`,
                position: "right",
              }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="technical-details">
        <div className="detail-item">
          <span>RSI (14):</span>
          <span
            className={
              stock.technical.rsi_14 < 30
                ? "value-oversold"
                : stock.technical.rsi_14 > 70
                ? "value-overbought"
                : ""
            }
          >
            {stock.technical.rsi_14?.toFixed(2)}
          </span>
        </div>
        <div className="detail-item">
          <span>Volume:</span>
          <span>{(stock.technical.volume / 1000000).toFixed(2)}M</span>
        </div>
        <div className="detail-item">
          <span>Rel Volume:</span>
          <span>{stock.technical.relative_volume?.toFixed(2)}</span>
        </div>
        <div className="detail-item">
          <span>ATR %:</span>
          <span>{stock.technical.atr_pct?.toFixed(2)}%</span>
        </div>
        <div className="detail-item">
          <span>Trend:</span>
          <span
            className={
              stock.technical.market_stage === "downtrend"
                ? "trend-down"
                : "trend-up"
            }
          >
            {stock.technical.market_stage}
          </span>
        </div>
      </div>
    </div>
  );
};

export default StockChart;
