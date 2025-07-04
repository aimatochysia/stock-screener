import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import { getStockCSV } from "../utils/github";
import { parseCSV } from "../utils/csvParser";

export const useTechnicalData = (ticker: string) => {
  return useQuery([ticker, "technical"], async () => {
    const url = getStockCSV(`${ticker}_technical.csv`);
    const { data } = await axios.get(url);
    return parseCSV(data);
  });
};
