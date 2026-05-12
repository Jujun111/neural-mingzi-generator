interface ResultListProps {
  emptyText: string;
  items: string[];
}

export function ResultList({ emptyText, items }: ResultListProps) {
  if (items.length === 0) {
    return (
      <div className="empty-state">
        <p>{emptyText}</p>
      </div>
    );
  }

  return (
    <ol className="result-list">
      {items.map((item, index) => (
        <li key={`${item}-${index}`} className="result-list__item">
          <span className="result-list__index">{String(index + 1).padStart(2, "0")}</span>
          <span className="result-list__name">{item}</span>
        </li>
      ))}
    </ol>
  );
}
