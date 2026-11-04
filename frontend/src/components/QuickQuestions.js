import React from "react";

const QUICK_QUESTIONS = {
  English: [
    {
      icon: "🌱",
      label: "Best crop this season?",
      query: "What is the best crop to grow in the current Rabi season for alluvial soil?",
    },
    {
      icon: "💰",
      label: "Tomato mandi rate",
      query: "What is today's mandi rate for tomatoes?",
    },
    {
      icon: "🌤",
      label: "Weather forecast",
      query: "What is the weather forecast for the next 3 days and what farming activities should I do?",
    },
    {
      icon: "🐛",
      label: "Wheat pest alert",
      query: "My wheat crop has yellow stripes on the leaves. What pest or disease is this and how do I treat it?",
    },
    {
      icon: "🌍",
      label: "Soil recommendation",
      query: "What crops are suitable for black cotton soil in Maharashtra?",
    },
    {
      icon: "📈",
      label: "Best price crop",
      query: "Which vegetables have the highest mandi prices right now in Delhi?",
    },
    {
      icon: "💊",
      label: "Organic pest control",
      query: "How can I control pests in my rice crop using organic methods?",
    },
    {
      icon: "🌾",
      label: "Wheat fertilizer",
      query: "What is the recommended fertilizer schedule for wheat at the tillering stage?",
    },
  ],
  Hindi: [
    {
      icon: "🌱",
      label: "इस मौसम में कौन सी फसल?",
      query: "रबी सीजन में काली मिट्टी में कौन सी फसल बोनी चाहिए?",
    },
    {
      icon: "💰",
      label: "टमाटर का मंडी भाव",
      query: "आज दिल्ली मंडी में टमाटर का क्या भाव है?",
    },
    {
      icon: "🌤",
      label: "मौसम की जानकारी",
      query: "अगले 3 दिनों का मौसम कैसा रहेगा? खेती के लिए क्या सलाह है?",
    },
    {
      icon: "🐛",
      label: "गेहूं में पीला रतुआ",
      query: "मेरे गेहूं की पत्तियों पर पीली धारियां आ गई हैं। यह कौन सी बीमारी है और इसका इलाज क्या है?",
    },
    {
      icon: "🌍",
      label: "मिट्टी की जानकारी",
      query: "मेरी जमीन में काली मिट्टी है, कौन सी फसल सबसे अच्छी होगी?",
    },
    {
      icon: "📈",
      label: "सबसे महंगी सब्जी",
      query: "अभी मंडी में कौन सी सब्जियों का भाव सबसे ज्यादा है?",
    },
    {
      icon: "💊",
      label: "जैविक कीट नियंत्रण",
      query: "धान की फसल में कीड़ों को जैविक तरीके से कैसे रोकें?",
    },
    {
      icon: "🌾",
      label: "MSP क्या है?",
      query: "इस साल गेहूं का MSP (न्यूनतम समर्थन मूल्य) क्या है?",
    },
  ],
};

/**
 * Preset quick question buttons for common farmer queries.
 */
export default function QuickQuestions({ language, onSelect, disabled }) {
  const questions = QUICK_QUESTIONS[language] || QUICK_QUESTIONS.English;

  return (
    <div className="w-full">
      <p className="text-xs text-gray-500 font-medium mb-2 text-center">
        {language === "Hindi"
          ? "⚡ जल्दी सवाल करें"
          : "⚡ Quick questions"}
      </p>
      <div className="flex flex-wrap gap-2 justify-center">
        {questions.map((q, i) => (
          <button
            key={i}
            onClick={() => onSelect(q.query)}
            disabled={disabled}
            title={q.query}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-full text-xs font-medium
              border transition-all duration-200 select-none
              ${disabled
                ? "opacity-40 cursor-not-allowed border-gray-200 text-gray-400 bg-white"
                : "border-farm-200 text-farm-700 bg-farm-50 hover:bg-farm-100 hover:border-farm-400 cursor-pointer active:scale-95"
              }`}
          >
            <span>{q.icon}</span>
            <span>{q.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
