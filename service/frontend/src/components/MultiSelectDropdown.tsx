import { useState, useRef, useEffect, useCallback } from 'react';
import { ChevronDown, Search, X, Check } from 'lucide-react';

interface MultiSelectProps {
  label: string;
  options: string[];
  selected: string[];
  onChange: (selected: string[]) => void;
  showSearch?: boolean;
  single?: boolean;
  hideLabel?: boolean;
  hideAllOption?: boolean;
}

export function MultiSelectDropdown({
  label,
  options,
  selected,
  onChange,
  showSearch = false,
  single = false,
  hideLabel = false,
  hideAllOption = false
}: MultiSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  // Thứ tự option CHỐT lúc mở: đã chọn lên đầu. Tick/bỏ tick khi đang mở không xáo lại list
  // dưới tay user — lần mở sau mới sắp lại.
  const [orderedOptions, setOrderedOptions] = useState<string[]>(options);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const closeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearCloseTimer = useCallback(() => {
    if (closeTimerRef.current) {
      clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
  }, []);

  const closeDropdown = useCallback(() => {
    setIsOpen(false);
    clearCloseTimer();
  }, [clearCloseTimer]);

  const openDropdown = () => {
    setOrderedOptions([
      ...options.filter(o => selected.includes(o)),
      ...options.filter(o => !selected.includes(o)),
    ]);
    setIsOpen(true);
  };

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        closeDropdown();
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [closeDropdown]);

  useEffect(() => clearCloseTimer, [clearCloseTimer]);

  // Rê chuột ra ngoài → đóng sau 150ms (rê nhầm qua mép thì không mất dropdown ngay)
  const handleMouseLeave = () => {
    if (!isOpen) return;
    clearCloseTimer();
    closeTimerRef.current = setTimeout(closeDropdown, 150);
  };

  const toggleOption = (option: string) => {
    if (single) {
      // Show Main/Show Cold luôn phải có 1 giá trị; filter 1-lựa-chọn thì bấm lại = bỏ chọn
      if (!label.startsWith('Show') && selected.includes(option)) {
        onChange([]);
      } else {
        onChange([option]);
      }
      closeDropdown();
    } else {
      if (selected.includes(option)) {
        onChange(selected.filter(item => item !== option));
      } else {
        onChange([...selected, option]);
      }
    }
  };

  const getOptionLabel = (option: string) => {
    if (label === 'Type' && option === '?') return 'Unknown';
    if (label.startsWith('Show')) return `Top ${option}`;
    return option;
  };

  const getButtonText = () => {
    if (selected.length === 0) return `All ${label}s`;
    if (single) return getOptionLabel(selected[0]);
    return `${selected.length} selected`;
  };

  const filteredOptions = showSearch
    ? orderedOptions.filter(opt => opt.toLowerCase().includes(searchQuery.toLowerCase()))
    : orderedOptions;

  const showAllOption = single && !label.startsWith('Show') && !hideAllOption;

  return (
    <div
      className="relative flex-1 min-w-0"
      ref={dropdownRef}
      onMouseEnter={clearCloseTimer}
      onMouseLeave={handleMouseLeave}
    >
      {!hideLabel && (
        <span className="block font-semibold text-gray-500 uppercase tracking-wider text-xs mb-1">
          {label}
        </span>
      )}
      <button
        type="button"
        onClick={() => (isOpen ? closeDropdown() : openDropdown())}
        className="w-full border flex items-center justify-between focus:outline-none focus:ring-1 focus:ring-gray-900 cursor-pointer transition-all duration-300 ease-in-out motion-reduce:transition-none px-3 py-2 text-sm border-gray-300 text-gray-900 bg-white hover:border-gray-400"
      >
        <span className="truncate flex items-center">
          <span className="truncate text-left">
            {getButtonText()}
          </span>
        </span>
        <ChevronDown className="w-3.5 h-3.5 ml-1 flex-shrink-0 text-gray-500" />
      </button>

      {isOpen && (
        <div className="absolute left-0 right-0 mt-1 bg-white border border-gray-300 shadow-md z-40 max-h-60 overflow-y-auto animate-in fade-in duration-100">
          {showSearch && (
            <div className="p-2 border-b border-gray-100 flex items-center bg-gray-50 sticky top-0 z-10">
              <Search className="w-3.5 h-3.5 text-gray-400 mr-2 flex-shrink-0" />
              <input
                type="text"
                placeholder={`Search ${label}...`}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full text-xs bg-transparent focus:outline-none placeholder-gray-400 text-gray-900"
              />
              {searchQuery && (
                <button
                  type="button"
                  onClick={() => setSearchQuery('')}
                  className="p-0.5 hover:bg-gray-200 text-gray-500 rounded-full"
                >
                  <X className="w-3 h-3" />
                </button>
              )}
            </div>
          )}
          <div className="py-1">
            {showAllOption && (
              <button
                type="button"
                onClick={() => {
                  onChange([]);
                  closeDropdown();
                }}
                className="w-full px-3 py-1.5 text-xs text-left text-gray-700 hover:bg-gray-100 flex items-center justify-between cursor-pointer transition-colors"
              >
                <span className="truncate">All {label}s</span>
                {selected.length === 0 && <Check className="w-3.5 h-3.5 text-gray-900 flex-shrink-0" />}
              </button>
            )}
            {filteredOptions.length === 0 ? (
              !showAllOption && <div className="px-3 py-2 text-xs text-gray-400">No options found</div>
            ) : (
              filteredOptions.map((option) => {
                const isSelected = selected.includes(option);
                const displayLabel = getOptionLabel(option);
                return (
                  <button
                    key={option}
                    type="button"
                    onClick={() => toggleOption(option)}
                    className="w-full px-3 py-1.5 text-xs text-left text-gray-700 hover:bg-gray-100 flex items-center justify-between cursor-pointer transition-colors"
                  >
                    <span className="truncate mr-2">{displayLabel}</span>
                    {isSelected && <Check className="w-3.5 h-3.5 text-gray-900 flex-shrink-0" />}
                  </button>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
}

