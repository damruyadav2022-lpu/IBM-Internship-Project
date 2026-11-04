import React, { useState } from "react";
import ChatInterface from "./components/ChatInterface";
import DiseaseDetector from "./components/DiseaseDetector";

const TABS = [
  { id: "chat", label: "💬 AI Advisor", labelHi: "💬 AI सलाहकार" },
  { id: "disease", label: "🔍 Disease Detection", labelHi: "🔍 रोग पहचान" },
];

export default function App() {
  const [language, setLanguage] = useState("English");
  const [location, setLocation] = useState("Delhi");
  const [activeTab, setActiveTab] = useState("chat");

  const isHindi = language === "Hindi";

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-farm-50 via-white to-farm-100">
      {/* Header */}
      <header className="bg-farm-700 text-white shadow-lg sticky top-0 z-20">
        <div className="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="text-3xl select-none">🌾</div>
            <div>
              <h1 className="text-xl font-bold leading-tight tracking-tight">
                KrishiRakshak
              </h1>
              <p className="text-farm-200 text-xs font-medium">
                Smart Farming Advice • कृषि सहायक
              </p>
            </div>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-3">
            {/* Location */}
            <div className="hidden sm:flex items-center gap-1.5 bg-farm-800 rounded-lg px-3 py-1.5">
              <span className="text-sm">📍</span>
              <select
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                className="bg-transparent text-white text-sm font-medium focus:outline-none cursor-pointer"
              >
                {["Delhi", "Mumbai", "Pune", "Lucknow", "Jaipur"].map((c) => (
                  <option key={c} value={c} className="text-gray-800">
                    {c}
                  </option>
                ))}
              </select>
            </div>

            {/* Language Toggle */}
            <div className="flex bg-farm-800 rounded-lg overflow-hidden text-sm font-medium">
              {["English", "Hindi"].map((lang) => (
                <button
                  key={lang}
                  onClick={() => setLanguage(lang)}
                  className={`px-3 py-1.5 transition-colors duration-200 ${
                    language === lang
                      ? "bg-white text-farm-700"
                      : "text-farm-200 hover:text-white"
                  }`}
                >
                  {lang === "English" ? "EN" : "हिं"}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Mobile location bar */}
        <div className="sm:hidden bg-farm-800 border-t border-farm-600 px-4 py-2">
          <div className="flex items-center gap-2">
            <span className="text-xs text-farm-300">📍 Location:</span>
            <select
              value={location}
              onChange={(e) => setLocation(e.target.value)}
              className="bg-transparent text-white text-xs font-medium focus:outline-none cursor-pointer"
            >
              {["Delhi", "Mumbai", "Pune", "Lucknow", "Jaipur"].map((c) => (
                <option key={c} value={c} className="text-gray-800">
                  {c}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Tab navigation */}
        <div className="bg-farm-800 border-t border-farm-600">
          <div className="max-w-4xl mx-auto px-4 flex">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2.5 text-sm font-medium transition-all duration-200 border-b-2 ${
                  activeTab === tab.id
                    ? "border-white text-white"
                    : "border-transparent text-farm-300 hover:text-farm-100 hover:border-farm-400"
                }`}
              >
                {isHindi ? tab.labelHi : tab.label}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 flex flex-col max-w-4xl w-full mx-auto px-4 py-4">
        {activeTab === "chat" && (
          <ChatInterface language={language} location={location} />
        )}
        {activeTab === "disease" && (
          <DiseaseDetector language={language} />
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-100 py-3">
        <div className="max-w-4xl mx-auto px-4 text-center text-xs text-gray-400">
          🌱 KrishiRakshak — AI-powered farming advice for Indian farmers •
          Data may not reflect real-time market conditions
        </div>
      </footer>
    </div>
  );
}
