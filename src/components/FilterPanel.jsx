import React from 'react';

const FilterPanel = ({
  filters,
  onFilterChange,
  onReset,
}) => {
  return (
    <div className="filter-panel">
      <h3>Filter & Screen</h3>
      
      <div className="filter-group">
        <label>RSI Range</label>
        <div className="filter-row">
          <input
            type="number"
            placeholder="Min"
            value={filters.rsiMin}
            onChange={(e) => onFilterChange('rsiMin', e.target.value)}
          />
          <span>-</span>
          <input
            type="number"
            placeholder="Max"
            value={filters.rsiMax}
            onChange={(e) => onFilterChange('rsiMax', e.target.value)}
          />
        </div>
        <div className="filter-presets">
          <button onClick={() => { onFilterChange('rsiMin', '0'); onFilterChange('rsiMax', '30'); }}>
            Oversold (&lt;30)
          </button>
          <button onClick={() => { onFilterChange('rsiMin', '70'); onFilterChange('rsiMax', '100'); }}>
            Overbought (&gt;70)
          </button>
        </div>
      </div>

      <div className="filter-group">
        <label>Volume (M)</label>
        <div className="filter-row">
          <input
            type="number"
            placeholder="Min"
            value={filters.volumeMin}
            onChange={(e) => onFilterChange('volumeMin', e.target.value)}
          />
        </div>
      </div>

      <div className="filter-group">
        <label>Price Range</label>
        <div className="filter-row">
          <input
            type="number"
            placeholder="Min"
            value={filters.priceMin}
            onChange={(e) => onFilterChange('priceMin', e.target.value)}
          />
          <span>-</span>
          <input
            type="number"
            placeholder="Max"
            value={filters.priceMax}
            onChange={(e) => onFilterChange('priceMax', e.target.value)}
          />
        </div>
      </div>

      <div className="filter-group">
        <label>Market Stage</label>
        <select
          value={filters.marketStage}
          onChange={(e) => onFilterChange('marketStage', e.target.value)}
        >
          <option value="">All</option>
          <option value="uptrend">Uptrend</option>
          <option value="downtrend">Downtrend</option>
          <option value="sideways">Sideways</option>
          <option value="neutral">Neutral</option>
        </select>
      </div>

      <div className="filter-group">
        <label>MA Alignment</label>
        <select
          value={filters.maAlignment}
          onChange={(e) => onFilterChange('maAlignment', e.target.value)}
        >
          <option value="">All</option>
          <option value="bullish">Bullish</option>
          <option value="bearish">Bearish</option>
          <option value="mixed">Mixed</option>
        </select>
      </div>

      <div className="filter-group">
        <label>% to SMA50</label>
        <div className="filter-row">
          <input
            type="number"
            placeholder="Min"
            value={filters.sma50Min}
            onChange={(e) => onFilterChange('sma50Min', e.target.value)}
          />
          <span>-</span>
          <input
            type="number"
            placeholder="Max"
            value={filters.sma50Max}
            onChange={(e) => onFilterChange('sma50Max', e.target.value)}
          />
        </div>
      </div>

      <div className="filter-group">
        <label>ATR % Range</label>
        <div className="filter-row">
          <input
            type="number"
            placeholder="Min"
            value={filters.atrMin}
            onChange={(e) => onFilterChange('atrMin', e.target.value)}
          />
          <span>-</span>
          <input
            type="number"
            placeholder="Max"
            value={filters.atrMax}
            onChange={(e) => onFilterChange('atrMax', e.target.value)}
          />
        </div>
      </div>

      <div className="filter-actions">
        <button className="btn-reset" onClick={onReset}>
          Reset All Filters
        </button>
      </div>
    </div>
  );
};

export default FilterPanel;
