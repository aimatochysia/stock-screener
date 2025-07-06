import React, { useMemo } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  createColumnHelper,
} from "@tanstack/react-table";
import { useInView } from "react-intersection-observer";

const StockTable = ({
  data,
  onRowClick,
  onSort,
  sortConfig,
  loadMore,
  hasMore,
  isLoading,
  selectedStock = null,
}) => {
  // Debug logging
  console.log("StockTable received data:", data);
  console.log("Data length:", data?.length);
  console.log("Is loading:", isLoading);

  const { ref, inView } = useInView({
    threshold: 0.1,
  });

  React.useEffect(() => {
    if (inView && hasMore && !isLoading) {
      loadMore();
    }
  }, [inView, hasMore, isLoading, loadMore]);

  const columnHelper = createColumnHelper();

  const columns = useMemo(
    () => [
      columnHelper.accessor("symbol", {
        header: "Symbol",
        cell: (info) => info.getValue(),
      }),
      columnHelper.accessor("technical.close", {
        header: "Price",
        cell: (info) =>
          info
            .getValue()
            ?.toLocaleString("en-US", { minimumFractionDigits: 2 }) || "N/A",
      }),
      columnHelper.accessor("technical.volume", {
        header: "Volume",
        cell: (info) => (info.getValue() / 1000000).toFixed(2) + "M",
      }),
      columnHelper.accessor("technical.relative_volume", {
        header: "Rel Volume",
        cell: (info) => info.getValue()?.toFixed(2) || "N/A",
      }),
      columnHelper.accessor("technical.rsi_14", {
        header: "RSI (14)",
        cell: (info) => {
          const value = info.getValue();
          let className = "";
          if (value < 30) className = "rsi-oversold";
          if (value > 70) className = "rsi-overbought";
          return <span className={className}>{value?.toFixed(2)}</span>;
        },
      }),
      columnHelper.accessor("technical.atr_pct", {
        header: "ATR %",
        cell: (info) => (info.getValue() * 100)?.toFixed(2) + "%",
      }),
      columnHelper.accessor("technical.market_stage", {
        header: "Trend",
        cell: (info) => {
          const value = info.getValue();
          let className = "";
          if (value === "downtrend") className = "trend-down";
          if (value === "uptrend") className = "trend-up";
          return <span className={className}>{value}</span>;
        },
      }),
    ],
    [columnHelper]
  );

  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    manualSorting: true,
    state: {
      sorting: [{ id: sortConfig.key, desc: sortConfig.direction === "desc" }],
    },
    onSortingChange: (updater) => {
      const newSorting =
        typeof updater === "function"
          ? updater([
              { id: sortConfig.key, desc: sortConfig.direction === "desc" },
            ])
          : updater;

      if (newSorting.length > 0) {
        onSort(newSorting[0].id);
      }
    },
  });

  return (
    <div className="stock-table">
      {/* Add debug info */}
      <div
        style={{
          padding: "10px",
          background: "#f0f0f0",
          marginBottom: "10px",
        }}
      >
        <strong>Debug Info:</strong> Data items: {data?.length || 0} | Loading:{" "}
        {isLoading ? "Yes" : "No"} | Has more: {hasMore ? "Yes" : "No"}
      </div>

      {data?.length === 0 && !isLoading && (
        <div className="no-data">No stocks to display</div>
      )}

      <table>
        <thead>
          {table.getHeaderGroups().map((headerGroup) => (
            <tr key={headerGroup.id}>
              {headerGroup.headers.map((header) => (
                <th
                  key={header.id}
                  onClick={header.column.getToggleSortingHandler()}
                  className={
                    sortConfig.key === header.column.id ? "active-sort" : ""
                  }
                >
                  {header.isPlaceholder
                    ? null
                    : flexRender(
                        header.column.columnDef.header,
                        header.getContext()
                      )}
                  {sortConfig.key === header.column.id && (
                    <span>{sortConfig.direction === "asc" ? " ↑" : " ↓"}</span>
                  )}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row) => (
            <tr
              key={row.id}
              onClick={() => onRowClick(row.original)}
              className={
                selectedStock?.symbol === row.original.symbol ? "selected" : ""
              }
            >
              {row.getVisibleCells().map((cell) => (
                <td key={cell.id}>
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      {/* Loading indicator at the bottom */}
      <div ref={ref} className="loader-trigger">
        {isLoading && <div className="loader">Loading...</div>}
        {!hasMore && <div className="end-message">No more stocks to load</div>}
      </div>
    </div>
  );
};

export default StockTable;
