import { useCallback, useState } from 'react';
import type { Tab, TabPrefs } from '../types';

export type PrefsPatch = Partial<TabPrefs> | ((prev: TabPrefs) => Partial<TabPrefs>);
export type UpdatePrefs = (patch: PrefsPatch) => void;

const DEFAULT_PREFS: TabPrefs = {
  genres: [],
  themes: [],
  studios: [],
  types: [],
  minScore: 0,
  mainK: 20,
  coldK: 10,
  sortBy: 'relevance',
  sortAsc: false,
};

/** All filter/sort/display-count settings, one object per tab. */
export function useTabPrefs(activeTab: Tab) {
  const [prefsByTab, setPrefsByTab] = useState<Record<Tab, TabPrefs>>({
    username: DEFAULT_PREFS,
    guest: DEFAULT_PREFS,
  });

  // Patch the active tab's prefs: takes an object or an updater (reads prev, avoids stale state).
  const updatePrefs = useCallback<UpdatePrefs>((patch) => {
    setPrefsByTab(prev => {
      const cur = prev[activeTab];
      const p = typeof patch === 'function' ? patch(cur) : patch;
      return { ...prev, [activeTab]: { ...cur, ...p } };
    });
  }, [activeTab]);

  // Patch a given tab's prefs (used to restore the sort from the URL on mount).
  const updateTabPrefs = useCallback((tab: Tab, patch: Partial<TabPrefs>) => {
    setPrefsByTab(prev => ({ ...prev, [tab]: { ...prev[tab], ...patch } }));
  }, []);

  return { prefs: prefsByTab[activeTab], updatePrefs, updateTabPrefs };
}
