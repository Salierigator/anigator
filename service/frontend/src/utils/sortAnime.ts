import type { AnimeItem, SortKey } from '../types';

// Client-side sort. 'relevance' keeps the backend order (reversed for asc); nulls go last.
export function sortAnimeItems(items: AnimeItem[], sortBy: SortKey, isAsc: boolean): AnimeItem[] {
  const sorted = [...items];

  if (sortBy === 'relevance') {
    // desc (down arrow, the default) = backend order (most relevant first); asc = reversed.
    if (isAsc) {
      sorted.reverse();
    }
    return sorted;
  }

  sorted.sort((a, b) => {
    let valA: string | number | null = null;
    let valB: string | number | null = null;

    if (sortBy === 'score') {
      valA = a.mal_score;
      valB = b.mal_score;
    } else if (sortBy === 'popularity') {
      valA = a.popularity;
      valB = b.popularity;
    } else if (sortBy === 'date') {
      valA = a.start_date;
      valB = b.start_date;
    }

    const isNullA = valA === null || valA === undefined;
    const isNullB = valB === null || valB === undefined;

    if (isNullA && isNullB) return 0;
    if (isNullA) return 1;
    if (isNullB) return -1;

    if (sortBy === 'date') {
      const cmp = (valA as string).localeCompare(valB as string);
      return isAsc ? cmp : -cmp;
    } else if (sortBy === 'popularity') {
      // popularity is a rank (1 = most popular), inverted relative to score: desc = most popular first.
      return isAsc ? (valB as number) - (valA as number) : (valA as number) - (valB as number);
    } else {
      return isAsc ? (valA as number) - (valB as number) : (valB as number) - (valA as number);
    }
  });

  return sorted;
}
