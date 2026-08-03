import { X } from 'lucide-react';
import type { TabPrefs } from '../types';
import type { UpdatePrefs } from '../hooks/useTabPrefs';

interface ActiveFilterChipsProps {
  prefs: TabPrefs;
  updatePrefs: UpdatePrefs;
}

export function ActiveFilterChips({ prefs, updatePrefs }: ActiveFilterChipsProps) {
  const hasActiveFilters =
    prefs.genres.length > 0 ||
    prefs.types.length > 0 ||
    prefs.themes.length > 0 ||
    prefs.studios.length > 0 ||
    prefs.minScore > 0;

  if (!hasActiveFilters) return null;

  const handleRemoveGenre = (genre: string) => {
    updatePrefs({ genres: prefs.genres.filter(g => g !== genre) });
  };

  const handleRemoveType = (type: string) => {
    updatePrefs({ types: prefs.types.filter(t => t !== type) });
  };

  const handleRemoveTheme = (theme: string) => {
    updatePrefs({ themes: prefs.themes.filter(t => t !== theme) });
  };

  const handleRemoveStudio = (studio: string) => {
    updatePrefs({ studios: prefs.studios.filter(s => s !== studio) });
  };

  const handleRemoveMinScore = () => {
    updatePrefs({ minScore: 0 });
  };

  const handleClearAll = () => {
    updatePrefs({
      genres: [],
      types: [],
      themes: [],
      studios: [],
      minScore: 0,
    });
  };

  return (
    <div className="flex flex-wrap items-center justify-center gap-2 py-2">
      {prefs.genres.map((genre) => (
        <span
          key={`genre-${genre}`}
          className="inline-flex items-center gap-1.5 px-2 py-0.5 bg-gray-100 text-gray-700 text-xs border border-gray-200 rounded-none font-medium"
        >
          <span>{genre}</span>
          <button
            type="button"
            onClick={() => handleRemoveGenre(genre)}
            aria-label={`Remove filter ${genre}`}
            className="text-gray-500 hover:text-gray-900 cursor-pointer focus:outline-none"
          >
            <X className="w-3 h-3" />
          </button>
        </span>
      ))}

      {prefs.types.map((type) => {
        const displayLabel = type === '?' ? 'Unknown' : type;
        return (
          <span
            key={`type-${type}`}
            className="inline-flex items-center gap-1.5 px-2 py-0.5 bg-gray-100 text-gray-700 text-xs border border-gray-200 rounded-none font-medium"
          >
            <span>{displayLabel}</span>
            <button
              type="button"
              onClick={() => handleRemoveType(type)}
              aria-label={`Remove filter ${displayLabel}`}
              className="text-gray-500 hover:text-gray-900 cursor-pointer focus:outline-none"
            >
              <X className="w-3 h-3" />
            </button>
          </span>
        );
      })}

      {prefs.themes.map((theme) => (
        <span
          key={`theme-${theme}`}
          className="inline-flex items-center gap-1.5 px-2 py-0.5 bg-gray-100 text-gray-700 text-xs border border-gray-200 rounded-none font-medium"
        >
          <span>{theme}</span>
          <button
            type="button"
            onClick={() => handleRemoveTheme(theme)}
            aria-label={`Remove filter ${theme}`}
            className="text-gray-500 hover:text-gray-900 cursor-pointer focus:outline-none"
          >
            <X className="w-3 h-3" />
          </button>
        </span>
      ))}

      {prefs.studios.map((studio) => (
        <span
          key={`studio-${studio}`}
          className="inline-flex items-center gap-1.5 px-2 py-0.5 bg-gray-100 text-gray-700 text-xs border border-gray-200 rounded-none font-medium"
        >
          <span>{studio}</span>
          <button
            type="button"
            onClick={() => handleRemoveStudio(studio)}
            aria-label={`Remove filter ${studio}`}
            className="text-gray-500 hover:text-gray-900 cursor-pointer focus:outline-none"
          >
            <X className="w-3 h-3" />
          </button>
        </span>
      ))}

      {prefs.minScore > 0 && (
        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 bg-gray-100 text-gray-700 text-xs border border-gray-200 rounded-none font-medium">
          <span>Score ≥ {prefs.minScore}</span>
          <button
            type="button"
            onClick={handleRemoveMinScore}
            aria-label="Remove minimum score filter"
            className="text-gray-500 hover:text-gray-900 cursor-pointer focus:outline-none"
          >
            <X className="w-3 h-3" />
          </button>
        </span>
      )}

      <button
        type="button"
        onClick={handleClearAll}
        className="text-xs text-gray-500 hover:text-gray-900 underline cursor-pointer ml-1 font-medium focus:outline-none"
      >
        Clear all
      </button>
    </div>
  );
}
