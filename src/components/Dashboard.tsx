import StockTable from "./StockTable";
import StockCard from "./StockCard";

export default function Dashboard() {
  return (
    <div className="p-4 grid grid-cols-1 lg:grid-cols-3 gap-4">
      <div className="lg:col-span-2">
        <StockTable />
      </div>
      <div className="space-y-4">
        <h2 className="text-xl font-semibold">Top Picks</h2>
        <StockCard ticker="BBNI.JK" />
        {/* Repeat with more tickers */}
      </div>
    </div>
  );
}
