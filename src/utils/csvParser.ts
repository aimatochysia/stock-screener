export function parseCSV(csv: string): any[] {
  const [headerLine, ...lines] = csv.trim().split("\n");
  const headers = headerLine.split(",");
  return lines.map(line => {
    const values = line.split(",");
    return headers.reduce((obj, key, i) => {
      obj[key] = isNaN(+values[i]) ? values[i] : +values[i];
      return obj;
    }, {} as Record<string, any>);
  });
}
