import { useEffect, useRef, useState } from 'react';
import { recommendAPI } from '../api';
import type { RecommendResponse, Tab } from '../types';

export type SearchParams = { username?: string; mal_ids?: number[] };
export type HandleSearch = (
  params: SearchParams,
  searchTab: Tab,
  skipUrlSync?: boolean,
  passedWatchedSet?: number[]
) => void;

interface Params {
  watchedSet: Set<number>;
}

/** Pool kết quả theo tab + loading/error + handleSearch (dedupe, abort request cũ). */
export function useRecommendations({ watchedSet }: Params) {
  const [loadingStates, setLoadingStates] = useState<Record<Tab, boolean>>({
    username: false,
    guest: false
  });
  const [slowLoadingStates, setSlowLoadingStates] = useState<Record<Tab, boolean>>({
    username: false,
    guest: false
  });
  const [error, setError] = useState<string | null>(null);
  const [tabPools, setTabPools] = useState<Record<Tab, RecommendResponse | null>>({
    username: null,
    guest: null
  });
  const [searchedUsername, setSearchedUsername] = useState<string>(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get('u') || '';
  });

  const lastSearchedParamsRef = useRef<{
    tab: Tab;
    username?: string;
    mal_ids?: number[];
    exclude_ids?: number[];
  } | null>(null);

  const searchAbortControllerRef = useRef<AbortController | null>(null);

  const handleSearch: HandleSearch = async (params, searchTab, skipUrlSync = false, passedWatchedSet) => {
    const exIds = passedWatchedSet || Array.from(watchedSet);

    // Duplicate request checking
    if (
      lastSearchedParamsRef.current &&
      lastSearchedParamsRef.current.tab === searchTab &&
      lastSearchedParamsRef.current.username === params.username &&
      JSON.stringify(lastSearchedParamsRef.current.mal_ids) === JSON.stringify(params.mal_ids) &&
      JSON.stringify(lastSearchedParamsRef.current.exclude_ids) === JSON.stringify(exIds)
    ) {
      return;
    }

    lastSearchedParamsRef.current = {
      tab: searchTab,
      username: params.username,
      mal_ids: params.mal_ids,
      exclude_ids: exIds
    };

    // Abort pending request
    if (searchAbortControllerRef.current) {
      searchAbortControllerRef.current.abort();
    }
    const abortController = new AbortController();
    searchAbortControllerRef.current = abortController;

    setLoadingStates(prev => ({ ...prev, [searchTab]: true }));
    setError(null);

    // Xoá pool cũ nếu là search username mới, hoặc tab chưa có pool nào
    if (searchTab === 'username' || !tabPools[searchTab]) {
      setTabPools(prev => ({ ...prev, [searchTab]: null }));
    }

    if (searchTab === 'username' && params.username) {
      setSearchedUsername(params.username);
    }

    if (!skipUrlSync) {
      const url = new URL(window.location.href);
      if (searchTab === 'username' && params.username) {
        url.search = `?u=${encodeURIComponent(params.username)}`;
      } else if (searchTab === 'guest' && params.mal_ids && params.mal_ids.length > 0) {
        url.search = `?ids=${params.mal_ids.join(',')}`;
        const wSet = passedWatchedSet ? new Set(passedWatchedSet) : watchedSet;
        if (wSet.size > 0) {
          const wArr = Array.from(wSet);
          if (url.search.length + wArr.join(',').length < 2000) {
            url.searchParams.set('watched', wArr.join(','));
          }
        }
      }
      window.history.replaceState(null, '', url.toString());
    }

    try {
      const response = await recommendAPI({
        ...params,
        exclude_ids: exIds.length > 0 ? exIds : undefined,
        top_k: 500,
        cold_k: 200,
        sfw: true
      }, abortController.signal);

      setTabPools(prev => ({ ...prev, [searchTab]: response }));
    } catch (err: unknown) {
      const axiosErr = err as { name?: string; response?: { data?: { detail?: unknown } } };
      if (axiosErr.name === 'CanceledError' || axiosErr.name === 'AbortError') {
        return;
      }
      // Lỗi thật → xoá dedupe ref để user retry được cùng params
      lastSearchedParamsRef.current = null;
      if (axiosErr.response?.data?.detail) {
        setError(typeof axiosErr.response.data.detail === 'string'
          ? axiosErr.response.data.detail
          : JSON.stringify(axiosErr.response.data.detail));
      } else {
        setError('An unexpected error occurred while connecting to the server.');
      }
    } finally {
      if (!abortController.signal.aborted) {
        setLoadingStates(prev => ({ ...prev, [searchTab]: false }));
      }
    }
  };

  /** Force dừng search đang chạy (backend treo/lỗi) → mở khoá form ngay. */
  const cancelSearch = (tab: Tab) => {
    if (searchAbortControllerRef.current) {
      searchAbortControllerRef.current.abort();
    }
    lastSearchedParamsRef.current = null;
    setLoadingStates(prev => ({ ...prev, [tab]: false }));
  };

  // Loading >8s → bật thêm dòng "server is waking up" (backend free-tier spin down)
  useEffect(() => {
    if (!loadingStates.username) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- hết loading thì tắt cờ
      setSlowLoadingStates(prev => ({ ...prev, username: false }));
      return;
    }
    const timer = setTimeout(() => {
      setSlowLoadingStates(prev => ({ ...prev, username: true }));
    }, 8000);
    return () => clearTimeout(timer);
  }, [loadingStates.username]);

  useEffect(() => {
    if (!loadingStates.guest) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- hết loading thì tắt cờ
      setSlowLoadingStates(prev => ({ ...prev, guest: false }));
      return;
    }
    const timer = setTimeout(() => {
      setSlowLoadingStates(prev => ({ ...prev, guest: true }));
    }, 8000);
    return () => clearTimeout(timer);
  }, [loadingStates.guest]);

  return { tabPools, loadingStates, slowLoadingStates, error, searchedUsername, handleSearch, cancelSearch };
}
