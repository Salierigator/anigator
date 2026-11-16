/* eslint-disable @typescript-eslint/no-explicit-any */
// Jikan details for the MODAL, fetched client-side: the synopsis/genres load moves to the user's
// browser and off the backend's MAL token rate limit. Only the modal uses it (click-triggered, once
// per open), so no queue is needed: a cache + in-flight dedup + a short backoff on 429.
// Displayed numbers (MAL Score) are NOT taken from here: the modal pins them to the card's MAL v2 values.
const CACHE = new Map<number, any>();
const inflight = new Map<number, Promise<any | null>>();
const JIKAN_BASE = 'https://api.jikan.moe/v4';

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Synchronous cache read (reopening the same anime shows it at once, no fetch). */
export function getCachedJikanDetail(malId: number): any | undefined {
  return CACHE.get(malId);
}

/**
 * Full detail object of one anime from Jikan. null = unavailable (network/429/outage), and the
 * caller (modal) falls back to the backend /api/anime/{id}. Failures are NOT cached, so they can be retried.
 */
export function fetchJikanDetail(malId: number): Promise<any | null> {
  const cached = CACHE.get(malId);
  if (cached !== undefined) return Promise.resolve(cached);

  const existing = inflight.get(malId);
  if (existing) return existing;

  const p = (async () => {
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        const res = await fetch(`${JIKAN_BASE}/anime/${malId}`);
        if (res.status === 429) {
          await sleep(1000); // Jikan rate limit (3 req/s): wait, then retry
          continue;
        }
        if (!res.ok) throw new Error(`status ${res.status}`);
        const json = await res.json();
        const data = json?.data ?? null;
        if (data) CACHE.set(malId, data);
        return data;
      } catch {
        if (attempt === 2) return null;
        await sleep(500);
      }
    }
    return null;
  })();

  inflight.set(malId, p);
  return p.finally(() => inflight.delete(malId));
}
