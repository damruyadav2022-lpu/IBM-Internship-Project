import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * Renders a single chat message bubble (user or assistant).
 */
export default function MessageBubble({ message }) {
  const isUser = message.role === "user";
  const isError = message.isError;

  return (
    <div
      className={`flex items-end gap-2 animate-slide-up ${
        isUser ? "flex-row-reverse" : "flex-row"
      }`}
    >
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-9 h-9 rounded-full flex items-center justify-center text-lg select-none
          ${isUser ? "bg-farm-100" : isError ? "bg-red-100" : "bg-farm-700"}`}
      >
        {isUser ? "👨‍🌾" : isError ? "⚠️" : "🤖"}
      </div>

      {/* Bubble */}
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 shadow-sm
          ${isUser
            ? "bg-farm-600 text-white rounded-br-none"
            : isError
            ? "bg-red-50 border border-red-200 text-red-800 rounded-bl-none"
            : "bg-white border border-gray-100 text-gray-800 rounded-bl-none"
          }`}
      >
        {/* Message content */}
        {isUser ? (
          <p className="text-sm leading-relaxed whitespace-pre-wrap">
            {message.content}
          </p>
        ) : (
          <div className="prose-farm text-sm">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          </div>
        )}

        {/* Sources */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="mt-3 pt-3 border-t border-gray-100">
            <p className="text-xs text-gray-400 font-medium mb-1.5">
              📚 Sources consulted:
            </p>
            <div className="flex flex-wrap gap-1.5">
              {message.sources.map((src, i) => (
                <span key={i} className="source-tag">
                  {sourceIcon(src)} {formatSourceName(src)}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Agent steps (collapsible) */}
        {!isUser && message.agent_steps && message.agent_steps.length > 0 && (
          <AgentSteps steps={message.agent_steps} />
        )}

        {/* Timestamp */}
        <p
          className={`text-xs mt-2 ${
            isUser ? "text-farm-200 text-right" : "text-gray-400"
          }`}
        >
          {message.timestamp}
        </p>
      </div>
    </div>
  );
}

function AgentSteps({ steps }) {
  const [open, setOpen] = React.useState(false);
  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1 text-xs text-gray-400 hover:text-farm-600 transition-colors"
      >
        <span>{open ? "▼" : "▶"}</span>
        <span>
          {open ? "Hide" : "Show"} reasoning ({steps.length} step
          {steps.length !== 1 ? "s" : ""})
        </span>
      </button>
      {open && (
        <div className="mt-2 space-y-2">
          {steps.map((step, i) => (
            <div
              key={i}
              className="bg-gray-50 border border-gray-100 rounded-lg p-2.5 text-xs"
            >
              <div className="flex items-center gap-1.5 mb-1">
                <span className="font-medium text-farm-700">
                  {toolIcon(step.tool)} {formatSourceName(step.tool)}
                </span>
              </div>
              <p className="text-gray-500 truncate">
                <span className="font-medium text-gray-600">Input: </span>
                {step.input}
              </p>
              {step.output && (
                <p className="text-gray-500 mt-1 line-clamp-2">
                  <span className="font-medium text-gray-600">Result: </span>
                  {step.output.slice(0, 150)}
                  {step.output.length > 150 ? "…" : ""}
                </p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function sourceIcon(src) {
  const icons = {
    weather_forecast: "🌤",
    crop_recommendation: "🌱",
    pest_disease_control: "🐛",
    mandi_price_checker: "💰",
    agricultural_knowledge_base: "📚",
  };
  return icons[src] || "📋";
}

function toolIcon(tool) {
  return sourceIcon(tool);
}

function formatSourceName(src) {
  return (
    {
      weather_forecast: "Weather Forecast",
      crop_recommendation: "Crop Calendar",
      pest_disease_control: "Pest Control",
      mandi_price_checker: "Mandi Prices",
      agricultural_knowledge_base: "Knowledge Base",
    }[src] || src.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
  );
}
