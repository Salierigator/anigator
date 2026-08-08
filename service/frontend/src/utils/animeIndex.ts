/** Search anime client-side — thay cho proxy MAL v2 qua /api/search.
 *
 * Index tĩnh ở public/anime-index.json (sinh bởi scripts/build_search_index.py), tải lazy
 * lúc user mở tab "Pick Favorites". Đổi lại 1 lần tải ~1.2 MB (gzip) thì search tức thì,
 * không phụ thuộc backend — vốn ngủ sau 15 phút trên free tier.
 */
import MiniSearch from 'minisearch';
import type { SearchResultItem } from '../types';

/** [mal_id, title, english, synonyms, type, year, score, members, poster, in_corpus] */
type Row = [number, string, string, string[], string, number, number, number, string, 0 | 1];

export type AnimeSearcher = (q: string, limit: number) => SearchResultItem[];

const POSTER_PREFIX = 'https://cdn.myanimelist.net/images/anime/';

/** Bỏ dấu (ō→o, ū→u) + thường hoá, để "yuru yuri" khớp "Yuru Yuri", "chuunibyou" khớp "Chūnibyō". */
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

/** Tải + dựng index (memo hoá: gọi nhiều lần chỉ chạy 1 lần). */
export function loadAnimeIndex() {
  if (!indexPromise) {
    indexPromise = build().catch(err => {
      indexPromise = null;                       // cho retry lần sau nếu mạng lỗi
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
    storeFields: [],                             // giữ Row ở byId, đỡ nhân đôi bộ nhớ
    processTerm: normalize,
    searchOptions: {
      prefix: true,                              // autocomplete: gõ dở vẫn ra
      fuzzy: 0.2,                                // chịu được typo 1 ký tự
      combineWith: 'AND',                        // "kimi no na" phải khớp cả 3 term
      boost: { title: 3, english: 3, syn: 1.5 },
      boostDocument: (id: number) => {
        const r = byId.get(id as number)!;
        // members^0.3: "gin" ra Gintama (1.1M) chứ không phải OVA lẻ khớp exact token.
        // 0.3 là đầu plateau đo trên 25 query mẫu (0.2→18/25, 0.3→21/25, tới 0.45 không đổi).
        // in_corpus=0 bị dìm nhưng vẫn hiện: user cần biết "model chưa có", không phải im lặng.
        return Math.pow(1 + r[7], 0.3) * (r[9] ? 1 : 0.6);
      },
    },
  });

  mini.addAll(rows.map(r => ({
    id: r[0], title: r[1], english: r[2], syn: r[3].join(' '),
  })));

  /** Không có bonus này thì sequel đè bản gốc: "attack on titan" ra Final Season,
   *  "code geass" ra R2. Khớp trọn tiêu đề > khớp phần đầu > khớp giữa chừng. */
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
      .slice(0, 50)                              // re-rank trong top 50, đủ rộng
      .map(hit => {
        const r = byId.get(hit.id as number)!;
        return { r, score: hit.score * exactness(r, nq) };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
      .map(h => toItem(h.r));
  };
}
