import { ArrowUp, ArrowDown } from 'lucide-react';
import { MultiSelectDropdown } from './MultiSelectDropdown';
import type { FacetOptions, SortKey, TabPrefs } from '../types';
import type { UpdatePrefs } from '../hooks/useTabPrefs';

interface Props {
  hasPool: boolean;
  facetOptions: FacetOptions;
  prefs: TabPrefs;
  updatePrefs: UpdatePrefs;
}

const sortOptionsMap: Record<SortKey, string> = {
  relevance: 'Relevance',
  score: 'MAL Score',
  popularity: 'Popularity',
  date: 'Release Date',
};

function getSortDirectionLabel(key: SortKey, isAsc: boolean) {
  switch (key) {
    case 'relevance':
      return isAsc ? 'Reversed Order' : 'Original Order';
    case 'score':
      return isAsc ? 'Lowest Score' : 'Highest Score';
    case 'popularity':
      return isAsc ? 'Least Popular' : 'Most Popular';
    case 'date':
      return isAsc ? 'Oldest First' : 'Newest First';
    default:
      return isAsc ? 'Ascending' : 'Descending';
  }
}

/**
 * Panel filter/sort/số hiển thị dưới form search. Nằm cố định trong flow ở đầu trang.
 */
export function FilterPanel({ hasPool, facetOptions, prefs, updatePrefs }: Props) {
  const handleSortByChange = (newSortBy: SortKey) => {
    updatePrefs({
      sortBy: newSortBy,
      sortAsc: false,
    });
  };

  const handleSortLabelSelect = (val: string[]) => {
    if (val.length > 0) {
      const selectedLabel = val[0];
      const selectedKey = Object.keys(sortOptionsMap).find(
        key => sortOptionsMap[key as SortKey] === selectedLabel
      ) as SortKey;
      if (selectedKey) handleSortByChange(selectedKey);
    }
  };

  if (!hasPool) return null;

  return (
    <div className="bg-gray-50 max-w-3xl mx-auto border border-gray-200 p-5 rounded-none">
      <div className="w-full flex flex-col gap-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 w-full">
          <MultiSelectDropdown
            label="Genre"
            options={facetOptions.genres}
            selected={prefs.genres}
            onChange={(val) => updatePrefs({ genres: val })}
          />
          <MultiSelectDropdown
            label="Type"
            options={facetOptions.types}
            selected={prefs.types}
            onChange={(val) => updatePrefs({ types: val })}
            single={true}
          />
          <MultiSelectDropdown
            label="Theme"
            options={facetOptions.themes}
            selected={prefs.themes}
            onChange={(val) => updatePrefs({ themes: val })}
          />
          <MultiSelectDropdown
            label="Studio"
            options={facetOptions.studios}
            selected={prefs.studios}
            onChange={(val) => updatePrefs({ studios: val })}
            showSearch={true}
          />
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 w-full">
          <div className="w-full">
            <div className="flex justify-between items-center mb-1.5">
              <span className="font-semibold uppercase tracking-wider text-xs text-gray-500">
                Score ≥
              </span>
              <span className="font-mono font-semibold bg-white border border-gray-200 text-gray-900 rounded-none px-2 py-0.5 text-xs">
                {prefs.minScore.toFixed(1)}
              </span>
            </div>
            <div className="flex items-center h-[38px] w-full">
              <input
                type="range"
                min="0"
                max="10"
                step="0.1"
                value={prefs.minScore}
                onChange={(e) => updatePrefs({ minScore: parseFloat(e.target.value) })}
                className="w-full h-1 bg-gray-200 appearance-none cursor-pointer accent-gray-900 focus:outline-none"
              />
            </div>
          </div>

          <div className="flex flex-col w-full">
            <span className="block font-semibold text-gray-500 uppercase tracking-wider text-xs mb-1">
              Sort
            </span>
            <div className="flex items-start gap-2">
              <div className="flex-1 min-w-0">
                <MultiSelectDropdown
                  label="Sort"
                  options={["Relevance", "MAL Score", "Popularity", "Release Date"]}
                  selected={[sortOptionsMap[prefs.sortBy]]}
                  onChange={handleSortLabelSelect}
                  single={true}
                  hideLabel={true}
                  hideAllOption={true}
                />
              </div>
              <button
                type="button"
                onClick={() => updatePrefs({ sortAsc: !prefs.sortAsc })}
                title={getSortDirectionLabel(prefs.sortBy, prefs.sortAsc)}
                aria-label={getSortDirectionLabel(prefs.sortBy, prefs.sortAsc)}
                className="h-[38px] w-[38px] flex-shrink-0 border border-gray-300 bg-white text-gray-500 hover:border-gray-400 hover:text-gray-900 focus:outline-none focus:ring-1 focus:ring-gray-900 flex items-center justify-center cursor-pointer transition-colors rounded-none"
              >
                {prefs.sortAsc ? <ArrowUp className="w-4 h-4" /> : <ArrowDown className="w-4 h-4" />}
              </button>
            </div>
          </div>

          <MultiSelectDropdown
            label="Show Main"
            options={["10", "20", "50", "100", "200"]}
            selected={[String(prefs.mainK)]}
            onChange={(val) => {
              if (val.length > 0) updatePrefs({ mainK: Number(val[0]) });
            }}
            single={true}
          />
          <MultiSelectDropdown
            label="Show Cold"
            options={["5", "10", "20", "50", "100"]}
            selected={[String(prefs.coldK)]}
            onChange={(val) => {
              if (val.length > 0) updatePrefs({ coldK: Number(val[0]) });
            }}
            single={true}
          />
        </div>
      </div>
    </div>
  );
}

