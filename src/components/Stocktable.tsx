import { useTechnicalData } from "../hooks/useStockData";
import { useState } from "react";

const tickers = ["BBNI.JK", "BBCA.JK", "TLKM.JK"]; // Load only top 50 or 100 initially

export default function StockTable() {
  const [sortKey, setSortKey] = useState("rsi_14");

  return (
    <div>
      <table className="table-auto w-full text-sm">
        <thead>
          <tr>
            <th>Ticker</th>
            <th onClick={() => setSortKey("rsi_14")}>RSI</th>
            <th onClick={() => setSortKey("volume")}>Volume</th>
          </tr>
        </thead>
        <tbody>
          {tickers.map(ticker => {
            const { data } = useTechnicalData(ticker);
            if (!data || !data.length) return null;
            const latest = data[data.length - 1];
            return (
              <tr key={ticker}>
                <td>{ticker}</td>
                <td>{latest.rsi_14}</td>
                <td>{latest.volume}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
