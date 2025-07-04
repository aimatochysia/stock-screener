import { useTechnicalData } from "../hooks/useStockData";
import { useEffect, useState } from "react";
import axios from "axios";
import { getStockCSV } from "../utils/github";
import { parseCSV } from "../utils/csvParser";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts";

export default function StockCard({ ticker }: { ticker: string }) {
  const { data: technicalData } = useTechnicalData(ticker);
  const [ohlc, setOhlc] = useState<any[]>([]);
  const [levels, setLevels] = useState<number[]>([]);
  const [channel, setChannel] = useState<any | null>(null);

  useEffect(() => {
    const fetchChartData = async () => {
      const [ohlcCSV, levelCSV, channelCSV] = await Promise.all([
        axios.get(getStockCSV(`${ticker}.csv`)),
        axios.get(getStockCSV(`${ticker}_levels.csv`)),
        axios.get(getStockCSV(`${ticker}_channel.csv`)),
      ]);
      setOhlc(parseCSV(ohlcCSV.data).slice(-60)); // last 60 days
      setLevels(parseCSV(levelCSV.data).map(d => +d.level_price));
      setChannel(parseCSV(channelCSV.data).slice(-1)[0]); // latest channel
    };
    fetchChartData();
  }, [ticker]);

  return (
    <div className="border p-2 rounded-xl shadow bg-white">
      <h3 className="text-lg font-semibold">{ticker}</h3>
      <LineChart width={300} height={200} data={ohlc}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="Date" />
        <YAxis />
        <Tooltip />
        <Line type="monotone" dataKey="Close" stroke="#8884d8" />
        {levels.map((lvl, i) => (
          <Line key={i} type="monotone" dataKey={() => lvl} stroke="red" dot={false} />
        ))}
        {channel && (
          <>
            <Line type="monotone" dataKey={() => channel.start_upper} stroke="blue" dot={false} />
            <Line type="monotone" dataKey={() => channel.end_upper} stroke="blue" dot={false} />
            <Line type="monotone" dataKey={() => channel.start_lower} stroke="green" dot={false} />
            <Line type="monotone" dataKey={() => channel.end_lower} stroke="green" dot={false} />
          </>
        )}
      </LineChart>
    </div>
  );
}
