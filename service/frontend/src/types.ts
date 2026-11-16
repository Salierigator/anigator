export interface AnimeItem {
  mal_id: number;
  title: string;
  type: string;
  year: number | null;
  mal_score: number | null;
  pred?: number | null;
  cos?: number | null;
  image_url: string | null;
  genres: string[];
  themes: string[];
  studios: string[];
  popularity: number | null;
  members: number | null;
  start_date: string | null;
}

export interface RecommendMeta {
  source: string;
  split: string;
  history_count: number;
  alpha: number | null;
  k_retrieve: number | null;
  mode: string;
  total_entries: number | null;
}

export interface RecommendRequest {
  username?: string;
  mal_ids?: number[];
  exclude_ids?: number[];
  top_k: number;
  cold_k: number;
  sfw?: boolean;
}

export interface RecommendResponse {
  main: AnimeItem[];
  cold: AnimeItem[];
  meta: RecommendMeta;
}

/** One entry of POST /api/posters: poster + fresh score/members (MAL v2), keyed by mal_id. */
export interface PosterEntry {
  poster: string | null;
  score: number | null;
  members: number | null;
}

export interface SearchResultItem {
  mal_id: number;
  title: string;
  title_english: string | null;
  type: string | null;
  year: number | null;
  mal_score: number | null;
  image_url: string | null;
  in_corpus: boolean;
}

// ---- UI state types (client-side only, not API shapes) ----

export type Tab = 'username' | 'guest';

export type SortKey = 'relevance' | 'score' | 'popularity' | 'date';

/** Filter / sort / display counts of EACH tab; changing them never calls the backend again. */
export interface TabPrefs {
  genres: string[];
  themes: string[];
  studios: string[];
  types: string[];
  minScore: number;
  mainK: number;
  coldK: number;
  sortBy: SortKey;
  sortAsc: boolean;
}

/** Available filter options, collected from the current pool. */
export interface FacetOptions {
  genres: string[];
  themes: string[];
  studios: string[];
  types: string[];
}
