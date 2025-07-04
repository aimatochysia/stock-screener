const GITHUB_RAW_BASE = "https://raw.githubusercontent.com";

export const getStockCSV = (filename: string) => {
  const repo = import.meta.env.VITE_STOCK_DB_REPO;
  const branch = import.meta.env.VITE_GITHUB_BRANCH;
  return `${GITHUB_RAW_BASE}/${repo}/${branch}/${filename}`;
};
