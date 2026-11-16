/** Client-side anime search for the guest picker.
 *
 * A prebuilt index in public/anime-index.json (made by scripts/build_search_index.py), loaded lazily
 * when the user opens the "Pick Favorites" tab. One ~1.2 MB (gzip) download buys instant search that
 * does not depend on the backend, which sleeps after 15 minutes on the free tier.
 */
import MiniSearch from 'minisearch';
import type { SearchResultItem } from '../types';

/** [mal_id, title, english, synonyms, type, year, score, members, poster, in_corpus] */
type Row = [number, string, string, string[], string, number, number, number, string, 0 | 1];

export type AnimeSearcher = (q: string, limit: number) => SearchResultItem[];

const POSTER_PREFIX = 'https://cdn.myanimelist.net/images/anime/';

/** Strip diacritics (ō -> o, ū -> u) and lowercase, so "yuru yuri" matches "Yuru Yuri" and "chunibyo" matches "Chūnibyō". */
const normalize = (term: string) =>
  term.normalize('NFKD').replace(/[\u0300-\u036f]/g, '').toLowerCase();

const toItem = (r: Row): SearchResultItem => ({
  mal_id: r[0],
  title: r[1],
  title_english: r[2] || null,
  type: r[4] || null,
  year: r[5] || null,
  mal_score: r[6] || null,
  image_url: r[8] ? `${POSTER_PREFIX}${r[8]}.jpg` : null,
  in_corpus: r[9] === 1,
});

let indexPromise: Promise<AnimeSearcher> | null = null;

/** Load and build the index (memoized: repeated calls run it once). */
export function loadAnimeIndex() {
  if (!indexPromise) {
    indexPromise = build().catch(err => {
      indexPromise = null;                       // allows a retry after a network error
      throw err;
    });
  }
  return indexPromise;
}

async function build(): Promise<AnimeSearcher> {
  const res = await fetch(`${import.meta.env.BASE_URL}anime-index.json`);
  if (!res.ok) throw new Error(`anime-index.json: HTTP ${res.status}`);
  const rows: Row[] = await res.json();

  const byId = new Map<number, Row>(rows.map(r => [r[0], r]));

  const mini = new MiniSearch<{ id: number; title: string; english: string; syn: string }>({
    fields: ['title', 'english', 'syn'],
    storeFields: [],                             // rows live in byId, so memory is not doubled
    processTerm: normalize,
    searchOptions: {
      prefix: true,                              // autocomplete: partial words match
      fuzzy: 0.2,                                // tolerates a one-character typo
      combineWith: 'AND',                        // "kimi no na" must match all 3 terms
      boost: { title: 3, english: 3, syn: 1.5 },
      boostDocument: (id: number) => {
        const r = byId.get(id as number)!;
        // members^0.3: "gin" ranks Gintama above an obscure OVA that matches the token exactly
        // (0.3 was tuned on sample queries; higher exponents gained nothing).
        // in_corpus=0 is pushed down but still shown: the user needs to see "not in the model", not silence.
        return Math.pow(1 + r[7], 0.3) * (r[9] ? 1 : 0.6);
      },
    },
  });

  mini.addAll(rows.map(r => ({
    id: r[0], title: r[1], english: r[2], syn: r[3].join(' '),
  })));

  /** Without this bonus, sequels beat the original: "attack on titan" gives Final Season,
   *  "code geass" gives R2. Whole-title match > prefix match > match in the middle. */
  const exactness = (r: Row, nq: string) => {
    for (const t of [r[1], r[2]]) {
      if (!t) continue;
      const nt = normalize(t);
      if (nt === nq) return 2.5;
      if (nt.startsWith(nq)) return 1.4;
    }
    return 1;
  };

  return (q: string, limit: number): SearchResultItem[] => {
    const nq = normalize(q);
    return mini.search(q)
      .slice(0, 50)                              // re-rank within the top 50, wide enough
      .map(hit => {
        const r = byId.get(hit.id as number)!;
        return { r, score: hit.score * exactness(r, nq) };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
      .map(h => toItem(h.r));
  };
}
