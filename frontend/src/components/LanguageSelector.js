import React from "react";

/**
 * Language selector toggle (English / Hindi).
 * Kept as a standalone component for reuse.
 */
export default function LanguageSelector({ language, onChange }) {
  const options = [
    { value: "English", label: "English", short: "EN", flag: "🇬🇧" },
    { value: "Hindi", label: "हिंदी", short: "हिं", flag: "🇮🇳" },
  ];

  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-500 font-medium hidden sm:inline">
        भाषा / Language:
      </span>
      <div className="flex bg-gray-100 rounded-lg p-0.5 gap-0.5">
        {options.map((opt) => (
          <button
            key={opt.value}
            onClick={() => onChange(opt.value)}
            title={opt.label}
            className={`flex items-center gap-1 px-3 py-1.5 rounded-md text-sm font-medium transition-all duration-200 ${
              language === opt.value
                ? "bg-farm-600 text-white shadow-sm"
                : "text-gray-600 hover:text-gray-800 hover:bg-white"
            }`}
          >
            <span>{opt.flag}</span>
            <span className="hidden sm:inline">{opt.label}</span>
            <span className="sm:hidden">{opt.short}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
