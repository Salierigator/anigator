import { useEffect } from 'react';
import type { Dispatch, RefObject, SetStateAction } from 'react';
import { fetchAnimeDetailAPI } from '../api';
import { GUEST_PICKS_KEY, WATCHED_SET_KEY } from './useGuestState';
import type { HandleSearch } from './useRecommendations';
import type { RecommendResponse, SearchResultItem, SortKey, Tab, TabPrefs } from '../types';

interface Params {
  hasInitializedRef: RefObject<boolean>;
  activeTab: Tab;
  setActiveTab: (tab: Tab) => void;
  tabPools: Record<Tab, RecommendResponse | null>;
  loadingStates: Record<Tab, boolean>;
  searchedUsername: string;
  guestPicks: SearchResultItem[];
  setGuestPicks: Dispatch<SetStateAction<SearchResultItem[]>>;
  watchedSet: Set<number>;
  setWatchedSet: Dispatch<SetStateAction<Set<number>>>;
  prefs: TabPrefs;
  updateTabPrefs: (tab: Tab, patch: Partial<TabPrefs>) => void;
  handleSearch: HandleSearch;
}

const SORT_KEYS = ['relevance', 'score', 'popularity', 'date'];

/**
 * 2 chiều URL ↔ state:
 * - Mount: đọc localStorage + query params (?u= / ?ids= / ?watched= / ?sort= / ?order=),
 *   khôi phục state và bắn search đầu tiên.
 * - Sau đó: state đổi → replaceState URL tương ứng (share link được).
 */
export function useUrlSync({
  hasInitializedRef,
  activeTab,
  setActiveTab,
  tabPools,
  loadingStates,
  searchedUsername,
  guestPicks,
  setGuestPicks,
  watchedSet,
  setWatchedSet,
  prefs,
  updateTabPrefs,
  handleSearch,
}: Params) {
  // Initialize from localStorage and URL
  useEffect(() => {
    if (hasInitializedRef.current) return;
    hasInitializedRef.current = true;

    // Load from localStorage first
    const savedPicks = localStorage.getItem(GUEST_PICKS_KEY);
    const savedWatched = localStorage.getItem(WATCHED_SET_KEY);
    let picks: SearchResultItem[] = savedPicks ? JSON.parse(savedPicks) : [];
    let watched: Set<number> = savedWatched ? new Set(JSON.parse(savedWatched)) : new Set();

    // Check URL params
    const params = new URLSearchParams(window.location.search);
    const uParam = params.get('u');
    const idsParam = params.get('ids');
    const watchedParam = params.get('watched');
    const sortParam = params.get('sort');
    const orderParam = params.get('order');

    if (uParam) {
      setActiveTab('username');
      if (watchedParam) {
        const wIds = watchedParam.split(',').map(Number).filter(n => !isNaN(n));
        watched = new Set(wIds);
        setWatchedSet(watched);
        localStorage.setItem(WATCHED_SET_KEY, JSON.stringify(Array.from(watched)));
      }
      if (sortParam && SORT_KEYS.includes(sortParam)) {
        updateTabPrefs('username', {
          sortBy: sortParam as SortKey,
          sortAsc: orderParam
            ? orderParam === 'asc'
            : false,
        });
      }
      handleSearch({ username: uParam }, 'username', true);
    } else if (idsParam) {
      setActiveTab('guest');
      const pickIds = idsParam.split(',').map(Number).filter(n => !isNaN(n));
      picks = pickIds.map(id => ({
        mal_id: id,
        title: `ID ${id}`,
        title_english: null,
        type: null,
        year: null,
        mal_score: null,
        image_url: null,
        in_corpus: true
      }));
      setGuestPicks(picks);
      localStorage.setItem(GUEST_PICKS_KEY, JSON.stringify(picks));

      if (watchedParam) {
        const wIds = watchedParam.split(',').map(Number).filter(n => !isNaN(n));
        watched = new Set(wIds);
        setWatchedSet(watched);
        localStorage.setItem(WATCHED_SET_KEY, JSON.stringify(Array.from(watched)));
      }
      if (sortParam && SORT_KEYS.includes(sortParam)) {
        updateTabPrefs('guest', {
          sortBy: sortParam as SortKey,
          sortAsc: orderParam
            ? orderParam === 'asc'
            : false,
        });
      }

      handleSearch({ mal_ids: pickIds }, 'guest', true, Array.from(watched));

      // Hydrate chips asynchronously
      pickIds.forEach(id => {
        fetchAnimeDetailAPI(id).then(res => {
          if (res) {
            setGuestPicks(prev => prev.map(p => p.mal_id === id ? {
              ...p,
              title: res.title,
              image_url: res.image_url,
              year: res.year,
              type: res.type
            } : p));
          }
        }).catch(() => {});
      });
    } else {
      setGuestPicks(picks);
      setWatchedSet(watched);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Synchronize URL parameters with active tab state
  useEffect(() => {
    if (!hasInitializedRef.current) return;

    const url = new URL(window.location.href);
    // Tab nào cũng chỉ giữ param của chính nó; "có pool HOẶC đang load" đều ghi (share link
    // được ngay khi vừa bấm search, chưa cần đợi kết quả).
    if (activeTab === 'username') {
      url.searchParams.delete('ids');
      url.searchParams.delete('watched');

      if ((tabPools.username || loadingStates.username) && searchedUsername) {
        url.searchParams.set('u', searchedUsername);
      } else {
        url.searchParams.delete('u');
      }
    } else {
      url.searchParams.delete('u');

      if ((tabPools.guest || loadingStates.guest) && guestPicks.length > 0) {
        url.searchParams.set('ids', guestPicks.map(p => p.mal_id).join(','));
        if (watchedSet.size > 0) {
          url.searchParams.set('watched', Array.from(watchedSet).join(','));
        } else {
          url.searchParams.delete('watched');
        }
      } else {
        url.searchParams.delete('ids');
        url.searchParams.delete('watched');
      }
    }

    // sort=relevance + order=desc là mặc định → bỏ khỏi URL cho gọn
    if (prefs.sortBy !== 'relevance' || prefs.sortAsc) {
      url.searchParams.set('sort', prefs.sortBy);
      url.searchParams.set('order', prefs.sortAsc ? 'asc' : 'desc');
    } else {
      url.searchParams.delete('sort');
      url.searchParams.delete('order');
    }

    window.history.replaceState(null, '', url.toString());
  }, [activeTab, tabPools.username, tabPools.guest, searchedUsername, guestPicks, watchedSet, loadingStates.username, loadingStates.guest, prefs.sortBy, prefs.sortAsc, hasInitializedRef]);
}
