import React, { useMemo } from 'react';

const MarketOverview = ({ stocks }) => {
  const stats = useMemo(() => {
    if (!stocks || stocks.length === 0) {
      return {
        total: 0,
        gainers: 0,
        losers: 0,
        neutral: 0,
        avgRSI: 0,
        avgVolume: 0,
        overbought: 0,
        oversold: 0,
        uptrend: 0,
        downtrend: 0,
      };
    }

    const gainers = stocks.filter(s => (s.sma_5_diff_pct || 0) > 0).length;
    const losers = stocks.filter(s => (s.sma_5_diff_pct || 0) < 0).length;
    const overbought = stocks.filter(s => (s.rsi_14 || 0) > 70).length;
    const oversold = stocks.filter(s => (s.rsi_14 || 0) < 30).length;
    const uptrend = stocks.filter(s => s.market_stage === 'uptrend').length;
    const downtrend = stocks.filter(s => s.market_stage === 'downtrend').length;

    const totalRSI = stocks.reduce((sum, s) => sum + (s.rsi_14 || 0), 0);
    const totalVolume = stocks.reduce((sum, s) => sum + (s.volume || 0), 0);

    return {
      total: stocks.length,
      gainers,
      losers,
      neutral: stocks.length - gainers - losers,
      avgRSI: stocks.length > 0 ? totalRSI / stocks.length : 0,
      avgVolume: stocks.length > 0 ? totalVolume / stocks.length : 0,
      overbought,
      oversold,
      uptrend,
      downtrend,
    };
  }, [stocks]);

  return (
    <div className="market-overview">
      <h3>Market Overview</h3>
      <div className="overview-grid">
        <div className="stat-card">
          <div className="stat-value">{stats.total}</div>
          <div className="stat-label">Total Stocks</div>
        </div>
        
        <div className="stat-card stat-positive">
          <div className="stat-value">{stats.gainers}</div>
          <div className="stat-label">Gainers</div>
        </div>
        
        <div className="stat-card stat-negative">
          <div className="stat-value">{stats.losers}</div>
          <div className="stat-label">Losers</div>
        </div>
        
        <div className="stat-card">
          <div className="stat-value">{stats.avgRSI.toFixed(1)}</div>
          <div className="stat-label">Avg RSI</div>
        </div>
        
        <div className="stat-card stat-positive">
          <div className="stat-value">{stats.uptrend}</div>
          <div className="stat-label">Uptrend</div>
        </div>
        
        <div className="stat-card stat-negative">
          <div className="stat-value">{stats.downtrend}</div>
          <div className="stat-label">Downtrend</div>
        </div>
        
        <div className="stat-card stat-positive">
          <div className="stat-value">{stats.oversold}</div>
          <div className="stat-label">Oversold</div>
        </div>
        
        <div className="stat-card stat-negative">
          <div className="stat-value">{stats.overbought}</div>
          <div className="stat-label">Overbought</div>
        </div>
      </div>
    </div>
  );
};

export default MarketOverview;
