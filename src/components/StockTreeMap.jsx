import React, { useMemo } from 'react';
import { Treemap, ResponsiveContainer, Tooltip } from 'recharts';

const StockTreeMap = ({ stocks }) => {
  const treemapData = useMemo(() => {
    if (!stocks || stocks.length === 0) return [];

    // Group stocks by market stage/trend
    const grouped = stocks.reduce((acc, stock) => {
      const trend = stock.market_stage || 'neutral';
      if (!acc[trend]) {
        acc[trend] = [];
      }
      acc[trend].push(stock);
      return acc;
    }, {});

    // Convert to treemap format
    const children = Object.entries(grouped).map(([trend, stocksList]) => ({
      name: trend,
      children: stocksList.map(stock => ({
        name: stock.symbol,
        size: Math.abs(stock.volume || 1000000) / 1000000, // Volume in millions
        value: stock.close || 0,
        rsi: stock.rsi_14,
        change: stock.sma_5_diff_pct || 0,
        trend: trend,
        volume: stock.volume,
        price: stock.close,
        atr_pct: stock.atr_pct,
      }))
    }));

    return [{
      name: 'Market',
      children: children
    }];
  }, [stocks]);

  const getColor = (entry) => {
    if (!entry) return '#757575';
    
    // Color based on price change
    const change = entry.change || 0;
    if (change > 5) return '#00C853'; // Strong green
    if (change > 2) return '#69F0AE'; // Light green
    if (change > 0) return '#B9F6CA'; // Very light green
    if (change > -2) return '#FFCDD2'; // Very light red
    if (change > -5) return '#EF5350'; // Light red
    return '#FF1744'; // Strong red
  };

  const CustomizedContent = (props) => {
    const { x, y, width, height, name, change, price } = props;

    if (width < 30 || height < 30) return null; // Don't render if too small

    return (
      <g>
        <rect
          x={x}
          y={y}
          width={width}
          height={height}
          style={{
            fill: getColor(props),
            stroke: '#fff',
            strokeWidth: 2,
            strokeOpacity: 0.8,
          }}
        />
        {width > 50 && height > 30 && (
          <>
            <text
              x={x + width / 2}
              y={y + height / 2 - 5}
              textAnchor="middle"
              fill="#fff"
              fontSize={Math.min(width / 5, 14)}
              fontWeight="bold"
            >
              {name}
            </text>
            {price && (
              <text
                x={x + width / 2}
                y={y + height / 2 + 10}
                textAnchor="middle"
                fill="#fff"
                fontSize={Math.min(width / 6, 10)}
              >
                {price.toFixed(0)}
              </text>
            )}
            {change !== undefined && (
              <text
                x={x + width / 2}
                y={y + height / 2 + 22}
                textAnchor="middle"
                fill="#fff"
                fontSize={Math.min(width / 6, 10)}
              >
                {change > 0 ? '+' : ''}{change.toFixed(1)}%
              </text>
            )}
          </>
        )}
      </g>
    );
  };

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload || !payload[0]) return null;

    const data = payload[0].payload;
    
    return (
      <div className="treemap-tooltip">
        <h4>{data.name}</h4>
        <p>Price: ${data.price?.toFixed(2) || 'N/A'}</p>
        <p>Change: {data.change > 0 ? '+' : ''}{data.change?.toFixed(2)}%</p>
        <p>Volume: {(data.volume / 1000000)?.toFixed(2) || 'N/A'}M</p>
        <p>RSI: {data.rsi?.toFixed(2) || 'N/A'}</p>
        <p>ATR%: {data.atr_pct?.toFixed(2) || 'N/A'}%</p>
        <p>Trend: {data.trend}</p>
      </div>
    );
  };

  if (!treemapData[0]?.children?.length) {
    return (
      <div className="no-data">
        No data available for treemap visualization
      </div>
    );
  }

  return (
    <div className="stock-treemap">
      <div className="treemap-legend">
        <div className="legend-item">
          <span className="legend-color" style={{ backgroundColor: '#00C853' }}></span>
          <span>Strong Up (&gt;5%)</span>
        </div>
        <div className="legend-item">
          <span className="legend-color" style={{ backgroundColor: '#69F0AE' }}></span>
          <span>Up (2-5%)</span>
        </div>
        <div className="legend-item">
          <span className="legend-color" style={{ backgroundColor: '#B9F6CA' }}></span>
          <span>Slight Up (0-2%)</span>
        </div>
        <div className="legend-item">
          <span className="legend-color" style={{ backgroundColor: '#FFCDD2' }}></span>
          <span>Slight Down (0 to -2%)</span>
        </div>
        <div className="legend-item">
          <span className="legend-color" style={{ backgroundColor: '#EF5350' }}></span>
          <span>Down (-2 to -5%)</span>
        </div>
        <div className="legend-item">
          <span className="legend-color" style={{ backgroundColor: '#FF1744' }}></span>
          <span>Strong Down (&lt;-5%)</span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height="100%">
        <Treemap
          data={treemapData}
          dataKey="size"
          aspectRatio={4 / 3}
          stroke="#fff"
          fill="#8884d8"
          content={<CustomizedContent />}
        >
          <Tooltip content={<CustomTooltip />} />
        </Treemap>
      </ResponsiveContainer>
    </div>
  );
};

export default StockTreeMap;
