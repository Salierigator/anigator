/* eslint-disable react-hooks/set-state-in-effect */
import { useState, useRef, useEffect } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import { Search, X, User, ExternalLink, HelpCircle, CheckCircle2, XCircle, Loader2, Check } from 'lucide-react';
import { checkUsernameExistsAPI } from '../api';
import { FilterPanel } from './FilterPanel';
import { loadAnimeIndex } from '../utils/animeIndex';
import type { AnimeSearcher } from '../utils/animeIndex';
import type { FacetOptions, SearchResultItem, Tab, TabPrefs } from '../types';
import type { UpdatePrefs } from '../hooks/useTabPrefs';

interface Props {
  onSubmit: (params: { username?: string; mal_ids?: number[] }, tab: Tab) => void;
  onCancel: () => void;
  isLoading: boolean;
  hasPool: boolean;
  activeTab: Tab;
  setActiveTab: (tab: Tab) => void;
  guestPicks: SearchResultItem[];
  setGuestPicks: Dispatch<SetStateAction<SearchResultItem[]>>;
  facetOptions: FacetOptions;
  prefs: TabPrefs;
  updatePrefs: UpdatePrefs;
}

export function SearchForm({
  onSubmit,
  onCancel,
  isLoading,
  hasPool,
  activeTab,
  setActiveTab,
  guestPicks,
  setGuestPicks,
  facetOptions,
  prefs,
  updatePrefs
}: Props) {
  const [username, setUsername] = useState(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get('u') || '';
  });
  const [lastSearchedUsername, setLastSearchedUsername] = useState(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get('u') || '';
  });

  const [usernameStatus, setUsernameStatus] = useState<'idle' | 'valid' | 'invalid'>('idle');
  const userAbortControllerRef = useRef<AbortController | null>(null);

  // Guest mode state
  const [guestQuery, setGuestQuery] = useState('');
  const [guestResults, setGuestResults] = useState<SearchResultItem[]>([]);
  const [showGuestDropdown, setShowGuestDropdown] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const [searchIndex, setSearchIndex] = useState<AnimeSearcher | null>(null);
  const [indexFailed, setIndexFailed] = useState(false);
  const searchDropdownRef = useRef<HTMLDivElement>(null);
  const itemRefs = useRef<Map<number, HTMLButtonElement>>(new Map());

  // Username validation
  useEffect(() => {
    if (activeTab !== 'username' || username.trim().length < 2) {
      setUsernameStatus('idle');
      if (userAbortControllerRef.current) userAbortControllerRef.current.abort();
      return;
    }

    if (userAbortControllerRef.current) userAbortControllerRef.current.abort();
    const abortController = new AbortController();
    userAbortControllerRef.current = abortController;

    const timer = setTimeout(async () => {
      try {
        const res = await checkUsernameExistsAPI(username, abortController.signal);
        if (res.exists) {
          setUsernameStatus('valid');
        } else {
          setUsernameStatus('invalid');
        }
      } catch (err: unknown) {
        const errorName = (err as { name?: string })?.name;
        if (errorName === 'CanceledError' || errorName === 'AbortError') return;
        setUsernameStatus('idle'); // Network error, status 502, stay silent
      }
    }, 600);
    return () => {
      clearTimeout(timer);
      abortController.abort();
    };
  }, [username, activeTab]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (searchDropdownRef.current && !searchDropdownRef.current.contains(event.target as Node)) {
        setShowGuestDropdown(false);
        setActiveIndex(-1);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Tải index lúc mở tab guest (~1.2 MB gzip, 1 lần / phiên) — không tải ở tab username.
  useEffect(() => {
    if (activeTab !== 'guest' || searchIndex) return;
    let alive = true;
    loadAnimeIndex().then(
      search => { if (alive) setSearchIndex(() => search); },
      err => { if (alive) { setIndexFailed(true); console.error('anime index', err); } }
    );
    return () => { alive = false; };
  }, [activeTab, searchIndex]);

  // Guest Search — chạy local, đồng bộ: không debounce, không abort, không cache.
  useEffect(() => {
    const q = guestQuery.trim();
    if (activeTab !== 'guest' || q.length < 2) {
      setGuestResults([]);
      setShowGuestDropdown(false);
      return;
    }
    setShowGuestDropdown(true);
    setGuestResults(searchIndex ? searchIndex(q, 10) : []);
  }, [guestQuery, activeTab, searchIndex]);

  const selectableItems = guestResults.filter(
    item => item.in_corpus && !guestPicks.some(p => p.mal_id === item.mal_id)
  );
  const indexLoading = activeTab === 'guest' && !searchIndex && !indexFailed;

  useEffect(() => {
    setActiveIndex(-1);
  }, [guestResults, showGuestDropdown]);

  useEffect(() => {
    if (activeIndex >= 0 && selectableItems[activeIndex]) {
      const malId = selectableItems[activeIndex].mal_id;
      const el = itemRefs.current.get(malId);
      if (el) {
        el.scrollIntoView({ block: 'nearest' });
      }
    }
  }, [activeIndex, selectableItems]);

  const addGuestPick = (item: SearchResultItem) => {
    if (!item.in_corpus) return;
    if (!guestPicks.some(p => p.mal_id === item.mal_id)) {
      setGuestPicks([...guestPicks, item]);
    }
    setGuestQuery('');
    setShowGuestDropdown(false);
    setActiveIndex(-1);
  };

  const removeGuestPick = (malId: number) => {
    setGuestPicks(guestPicks.filter(p => p.mal_id !== malId));
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown') {
      if (!showGuestDropdown && guestQuery.length >= 2 && guestResults.length > 0) {
        setShowGuestDropdown(true);
      }
      if (selectableItems.length > 0) {
        e.preventDefault();
        setActiveIndex(prev => (prev < selectableItems.length - 1 ? prev + 1 : 0));
      }
    } else if (e.key === 'ArrowUp') {
      if (selectableItems.length > 0 && showGuestDropdown) {
        e.preventDefault();
        setActiveIndex(prev => (prev > 0 ? prev - 1 : selectableItems.length - 1));
      }
    } else if (e.key === 'Enter') {
      if (showGuestDropdown && activeIndex >= 0 && activeIndex < selectableItems.length) {
        e.preventDefault();
        addGuestPick(selectableItems[activeIndex]);
      }
    } else if (e.key === 'Escape') {
      setShowGuestDropdown(false);
      setActiveIndex(-1);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (activeTab === 'username') {
      if (username.trim()) {
        setLastSearchedUsername(username.trim());
        onSubmit({ username: username.trim() }, 'username');
      }
    } else {
      if (guestPicks.length > 0) {
        onSubmit({ mal_ids: guestPicks.map(p => p.mal_id) }, 'guest');
      }
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-3xl mx-auto space-y-6">
      {/* Tabs */}
      <div className="flex justify-center mb-6">
        <div className="inline-flex bg-gray-100 p-1 border border-gray-200">
          <button
            type="button"
            onClick={() => setActiveTab('username')}
            className={`px-6 py-2 text-sm font-semibold transition-colors rounded-none cursor-pointer ${
              activeTab === 'username' ? 'bg-white text-gray-900 shadow-sm border border-gray-200' : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            MAL Username
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('guest')}
            className={`px-6 py-2 text-sm font-semibold transition-colors rounded-none cursor-pointer ${
              activeTab === 'guest' ? 'bg-white text-gray-900 shadow-sm border border-gray-200' : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            Pick Favorites
          </button>
        </div>
      </div>

      <div className="flex gap-2 max-w-xl mx-auto w-full">
        <div className="relative flex-1" ref={searchDropdownRef}>
          {activeTab === 'username' ? (
            <>
              <div className="absolute inset-y-0 left-0 flex items-center z-10 pointer-events-none">
                {username === lastSearchedUsername && lastSearchedUsername.length > 0 ? (
                  <a
                    href={`https://myanimelist.net/profile/${username}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="h-full flex items-center px-4 hover:bg-gray-100 transition-colors pointer-events-auto border-r border-transparent hover:border-gray-200 group"
                    title="View MAL Profile"
                  >
                    <ExternalLink className="h-5 w-5 text-gray-400 group-hover:text-gray-900 transition-colors" />
                  </a>
                ) : (
                  <div className="h-full flex items-center px-4">
                    <User className="h-5 w-5 text-gray-400" />
                  </div>
                )}
              </div>

              <input
                type="text"
                value={username}
                onChange={(e) => {
                  setUsername(e.target.value);
                  setUsernameStatus('idle');
                }}
                placeholder="Enter MAL Username..."
                className={`w-full ${(username === lastSearchedUsername && lastSearchedUsername.length > 0) ? 'pl-[3.25rem]' : 'pl-11'} pr-10 py-3 bg-white border border-gray-300 text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-gray-900 focus:border-gray-900 sm:text-base transition-shadow shadow-sm rounded-none`}
              />

              <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
                {usernameStatus === 'valid' && <CheckCircle2 className="w-5 h-5 text-green-500" />}
                {usernameStatus === 'invalid' && (
                  <div className="group relative">
                    <XCircle className="w-5 h-5 text-red-500 pointer-events-auto cursor-help" />
                    <div className="hidden group-hover:block absolute right-0 top-6 w-32 bg-gray-900 text-white text-[10px] p-1.5 z-10 text-center rounded-sm">
                      Username not found
                    </div>
                  </div>
                )}
              </div>
            </>
          ) : (
            <>
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <Search className="h-5 w-5 text-gray-400" />
              </div>
              <input
                type="text"
                role="combobox"
                aria-expanded={showGuestDropdown}
                aria-controls="guest-search-listbox"
                aria-activedescendant={
                  showGuestDropdown && activeIndex >= 0 && selectableItems[activeIndex]
                    ? `guest-option-${selectableItems[activeIndex].mal_id}`
                    : undefined
                }
                aria-autocomplete="list"
                value={guestQuery}
                onChange={(e) => {
                  setGuestQuery(e.target.value);
                  if (e.target.value.length >= 2) setShowGuestDropdown(true);
                }}
                onFocus={() => {
                  if (guestQuery.length >= 2 && guestResults.length > 0) setShowGuestDropdown(true);
                }}
                onKeyDown={handleKeyDown}
                placeholder="Search anime to add..."
                className="w-full pl-11 pr-10 py-3 bg-white border border-gray-300 text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-gray-900 focus:border-gray-900 sm:text-base transition-shadow shadow-sm rounded-none"
              />
              <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
                {indexLoading && <Loader2 className="w-4 h-4 text-gray-400 animate-spin" />}
              </div>

              {/* Guest Search Dropdown */}
              {showGuestDropdown && (
                <div
                  id="guest-search-listbox"
                  role="listbox"
                  aria-busy={indexLoading}
                  className="absolute top-full left-0 right-0 mt-1 bg-white border border-gray-300 shadow-xl z-50 max-h-80 overflow-y-auto"
                >
                  {guestResults.map(item => {
                    const isAlreadyPicked = guestPicks.some(p => p.mal_id === item.mal_id);
                    const isDisabled = !item.in_corpus || isAlreadyPicked;
                    const isHighlighted = showGuestDropdown && activeIndex >= 0 && selectableItems[activeIndex]?.mal_id === item.mal_id;

                    return (
                      <button
                        key={item.mal_id}
                        id={`guest-option-${item.mal_id}`}
                        type="button"
                        role="option"
                        aria-selected={isHighlighted}
                        ref={(el) => {
                          if (el) itemRefs.current.set(item.mal_id, el);
                          else itemRefs.current.delete(item.mal_id);
                        }}
                        onPointerDown={(e) => {
                          e.preventDefault();
                          if (!isDisabled) addGuestPick(item);
                        }}
                        disabled={isDisabled}
                        className={`w-full flex items-center gap-3 p-2 text-left border-b border-gray-100 last:border-0 transition-colors ${
                          isDisabled
                            ? 'opacity-50 cursor-not-allowed bg-gray-50'
                            : isHighlighted
                            ? 'bg-gray-100 cursor-pointer'
                            : 'hover:bg-gray-50 cursor-pointer'
                        }`}
                      >
                        {item.image_url ? (
                          <img src={item.image_url} alt="" className="w-10 h-14 object-cover flex-shrink-0 bg-gray-200" />
                        ) : (
                          <div className="w-10 h-14 bg-gray-200 flex-shrink-0" />
                        )}
                        <div className="flex-1 min-w-0">
                          <div className="font-semibold text-gray-900 truncate">{item.title}</div>
                          {item.title_english && <div className="text-xs text-gray-500 truncate">{item.title_english}</div>}
                          <div className="text-[10px] text-gray-400 mt-1 uppercase flex items-center gap-2">
                            <span>{item.type || '?'} {item.year ? `· ${item.year}` : ''}</span>
                            {item.mal_score && <span>★ {item.mal_score}</span>}
                          </div>
                        </div>
                        {!item.in_corpus && (
                          <div className="flex-shrink-0 px-2 group relative">
                            <HelpCircle className="w-4 h-4 text-gray-400" />
                            <div className="hidden group-hover:block absolute right-0 top-6 w-32 bg-gray-900 text-white text-[10px] p-1.5 z-10 text-center pointer-events-none">
                              Not in model yet
                            </div>
                          </div>
                        )}
                        {isAlreadyPicked && (
                          <div className="flex-shrink-0 px-2">
                            <Check className="w-4 h-4 text-gray-500" />
                          </div>
                        )}
                      </button>
                    );
                  })}
                  {indexLoading && guestResults.length === 0 && (
                    <div aria-hidden="true" className="animate-pulse">
                      {[...Array(5)].map((_, i) => (
                        <div
                          key={i}
                          className="w-full flex items-center gap-3 p-2 border-b border-gray-100 last:border-0"
                        >
                          <div className="w-10 h-14 bg-gray-200 flex-shrink-0" />
                          <div className="flex-1 min-w-0">
                            <div className="h-3 bg-gray-200 w-2/3" />
                            <div className="h-2 bg-gray-100 w-1/3 mt-2" />
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                  {/* search local nên "không có" là kết luận chắc chắn + tức thì, không
                      còn mơ hồ giữa "chưa về" và "không có" như hồi gọi API */}
                  {!indexLoading && !indexFailed && guestResults.length === 0 && (
                    <div className="p-3 text-sm text-gray-400 text-center">No anime matches “{guestQuery.trim()}”</div>
                  )}
                  {indexFailed && (
                    <div className="p-3 text-sm text-gray-500 text-center">
                      Can't load the anime index — check your connection and reload.
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
        {isLoading ? (
          <button
            key="stop"
            type="button"
            onClick={(e) => {
              // preventDefault: khi state đổi giữa lúc dispatch, React morph node này thành
              // nút submit → default action của click sẽ submit form ngay sau cancel
              e.preventDefault();
              onCancel();
            }}
            title="Stop the current search"
            className="w-48 py-3 bg-white border border-gray-900 text-gray-900 font-medium hover:bg-gray-100 focus:outline-none focus:ring-1 focus:ring-gray-900 transition-colors cursor-pointer rounded-none whitespace-nowrap"
          >
            Stop
          </button>
        ) : (
          <button
            key="search"
            type="submit"
            disabled={activeTab === 'username' ? !username.trim() : guestPicks.length === 0}
            className="w-48 py-3 bg-gray-900 text-white font-medium hover:bg-gray-800 focus:outline-none focus:ring-1 focus:ring-gray-900 disabled:opacity-50 disabled:cursor-not-allowed transition-colors cursor-pointer rounded-none whitespace-nowrap"
          >
            {activeTab === 'username' ? 'Get Goods :)' : 'Git Gud :)'}
          </button>
        )}
      </div>

      {activeTab === 'guest' && (
        <div className="max-w-xl mx-auto w-full space-y-2">
          {guestPicks.length === 0 ? (
            <div className="text-center text-sm text-gray-400">Search and pick at least 1 anime. (Recommended: 5+)</div>
          ) : (
            <>
              <div className="flex items-center justify-between text-xs">
                <span className="text-gray-500">{guestPicks.length} selected</span>
                <button
                  type="button"
                  onClick={() => setGuestPicks([])}
                  className="text-gray-500 hover:text-gray-900 underline underline-offset-2 cursor-pointer transition-colors"
                >
                  Clear all
                </button>
              </div>
              {/* ô đều nhau + click cả chip = xoá: nút X 14px dịch chỗ sau mỗi lần xoá thì
                  không spam click liên tục được */}
              <div className="grid grid-cols-2 sm:grid-cols-3 w-full gap-2">
                {guestPicks.map(pick => (
                  <button
                    key={pick.mal_id}
                    type="button"
                    onClick={() => removeGuestPick(pick.mal_id)}
                    title={`Remove ${pick.title}`}
                    aria-label={`Remove ${pick.title}`}
                    className="flex items-center w-full bg-white border border-gray-300 pr-2 overflow-hidden h-10 group cursor-pointer hover:border-gray-900 hover:bg-gray-50 transition-colors"
                  >
                    {pick.image_url ? (
                      <img src={pick.image_url} alt="" className="h-full w-7 object-cover mr-2 flex-shrink-0" />
                    ) : (
                      <div className="h-full w-7 bg-gray-200 mr-2 flex-shrink-0" />
                    )}
                    <span className="flex-1 min-w-0 text-xs font-medium truncate text-left mr-2">{pick.title}</span>
                    <X className="w-3.5 h-3.5 flex-shrink-0 text-gray-400 group-hover:text-gray-900 transition-colors" />
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      )}

      <FilterPanel
        hasPool={hasPool}
        facetOptions={facetOptions}
        prefs={prefs}
        updatePrefs={updatePrefs}
      />
    </form>
  );
}
