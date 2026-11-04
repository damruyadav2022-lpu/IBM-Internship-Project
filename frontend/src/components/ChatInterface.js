import React, { useState, useEffect, useRef, useCallback } from "react";
import MessageBubble from "./MessageBubble";
import QuickQuestions from "./QuickQuestions";
import { sendMessage, checkHealth } from "../services/api";

const WELCOME_MESSAGES = {
  English: {
    content:
      "Namaste! 🙏 I'm **KrishiRakshak**, your AI farming assistant.\n\n" +
      "I can help you with:\n" +
      "- 🌤 **Weather forecasts** and farming advisories\n" +
      "- 🌱 **Crop recommendations** by season and soil type\n" +
      "- 🐛 **Pest & disease** identification and treatment\n" +
      "- 💰 **Mandi prices** for vegetables, grains, and commodities\n" +
      "- 🌍 **Soil health** guidance and fertilizer advice\n\n" +
      "Ask me anything about farming, or try one of the quick questions below!",
    sources: [],
    agent_steps: [],
  },
  Hindi: {
    content:
      "नमस्ते! 🙏 मैं **KrishiRakshak** हूँ, आपका AI कृषि सहायक।\n\n" +
      "मैं आपकी इन विषयों में मदद कर सकता हूँ:\n" +
      "- 🌤 **मौसम की जानकारी** और खेती की सलाह\n" +
      "- 🌱 **फसल की सिफारिश** मौसम और मिट्टी के अनुसार\n" +
      "- 🐛 **कीट और रोग** पहचान और उपचार\n" +
      "- 💰 **मंडी भाव** सब्जियों, अनाज और जिंसों के लिए\n" +
      "- 🌍 **मिट्टी की जांच** और खाद की सलाह\n\n" +
      "खेती के बारे में कुछ भी पूछें, या नीचे दिए गए जल्दी सवालों में से चुनें!",
    sources: [],
    agent_steps: [],
  },
};

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/**
 * Main chat interface: manages message history, sends to API, shows bubbles.
 */
export default function ChatInterface({ language, location }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState("checking"); // "ok" | "error" | "checking"
  const bottomRef = useRef(null);
  const inputRef = useRef(null);
  const prevLanguage = useRef(language);

  // Add welcome message on mount and when language changes
  useEffect(() => {
    const welcome = WELCOME_MESSAGES[language] || WELCOME_MESSAGES.English;
    const welcomeMsg = {
      id: Date.now(),
      role: "assistant",
      content: welcome.content,
      sources: welcome.sources,
      agent_steps: welcome.agent_steps,
      timestamp: formatTime(new Date()),
    };
    if (prevLanguage.current !== language) {
      // Language changed — replace last welcome or prepend new one
      setMessages((prev) => {
        const filtered = prev.filter((m) => m.id !== prev[0]?.id);
        return [welcomeMsg, ...filtered];
      });
      prevLanguage.current = language;
    } else {
      setMessages([welcomeMsg]);
    }
  }, [language]); // eslint-disable-line react-hooks/exhaustive-deps

  // Backend health check on mount
  useEffect(() => {
    checkHealth()
      .then(() => setBackendStatus("ok"))
      .catch(() => setBackendStatus("error"));
  }, []);

  // Auto-scroll to bottom when messages update
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleSend = useCallback(
    async (messageText) => {
      const text = (messageText || input).trim();
      if (!text || loading) return;

      const userMsg = {
        id: Date.now(),
        role: "user",
        content: text,
        timestamp: formatTime(new Date()),
      };
      setMessages((prev) => [...prev, userMsg]);
      setInput("");
      setLoading(true);

      try {
        const data = await sendMessage(text, language, location);
        const assistantMsg = {
          id: Date.now() + 1,
          role: "assistant",
          content: data.response,
          sources: data.sources || [],
          agent_steps: data.agent_steps || [],
          timestamp: formatTime(new Date()),
        };
        setMessages((prev) => [...prev, assistantMsg]);
      } catch (err) {
        const errorMsg = {
          id: Date.now() + 1,
          role: "assistant",
          isError: true,
          content:
            "⚠️ **Connection Error**\n\n" +
            "Could not reach the KrishiRakshak backend. Please check:\n" +
            "1. Backend is running on `http://localhost:8000`\n" +
            "2. Your LLM is configured in `.env`\n" +
            "3. Run `docker-compose up` or `uvicorn main:app` in the backend folder.\n\n" +
            `Error details: ${err.message || "Unknown error"}`,
          sources: [],
          agent_steps: [],
          timestamp: formatTime(new Date()),
        };
        setMessages((prev) => [...prev, errorMsg]);
      } finally {
        setLoading(false);
        setTimeout(() => inputRef.current?.focus(), 100);
      }
    },
    [input, loading, language, location]
  );

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClear = () => {
    const welcome = WELCOME_MESSAGES[language] || WELCOME_MESSAGES.English;
    setMessages([
      {
        id: Date.now(),
        role: "assistant",
        content: welcome.content,
        sources: [],
        agent_steps: [],
        timestamp: formatTime(new Date()),
      },
    ]);
  };

  return (
    <div className="flex flex-col h-[calc(100vh-140px)] min-h-[500px]">
      {/* Backend status bar */}
      {backendStatus === "error" && (
        <div className="bg-red-50 border border-red-200 text-red-700 text-xs px-4 py-2 rounded-xl mb-3 flex items-center gap-2">
          <span>⚠️</span>
          <span>
            Backend not reachable. Ensure the server is running on port 8000.
          </span>
        </div>
      )}
      {backendStatus === "ok" && (
        <div className="bg-farm-50 border border-farm-200 text-farm-700 text-xs px-4 py-2 rounded-xl mb-3 flex items-center gap-2">
          <span className="w-2 h-2 bg-farm-500 rounded-full animate-pulse-slow inline-block" />
          <span>
            Connected to KrishiRakshak AI • Location:{" "}
            <span className="font-semibold">{location}</span>
          </span>
        </div>
      )}

      {/* Message list */}
      <div className="flex-1 overflow-y-auto space-y-4 py-2 px-1 scroll-smooth">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {/* Loading indicator */}
        {loading && (
          <div className="flex items-end gap-2 animate-fade-in">
            <div className="w-9 h-9 rounded-full bg-farm-700 flex items-center justify-center text-lg select-none">
              🤖
            </div>
            <div className="bg-white border border-gray-100 rounded-2xl rounded-bl-none px-5 py-4 shadow-sm">
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-400 mr-1">Thinking</span>
                <span className="typing-dot" />
                <span className="typing-dot" />
                <span className="typing-dot" />
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Quick questions */}
      <div className="pt-3 pb-2">
        <QuickQuestions
          language={language}
          onSelect={(q) => handleSend(q)}
          disabled={loading}
        />
      </div>

      {/* Input area */}
      <div className="pt-2">
        <div className="card p-3 flex items-end gap-3">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
            rows={1}
            placeholder={
              language === "Hindi"
                ? "अपना सवाल यहाँ लिखें... (Enter दबाएं)"
                : "Ask your farming question… (Enter to send)"
            }
            className={`input-field resize-none overflow-hidden min-h-[44px] max-h-32 py-2.5 text-sm flex-1 ${
              language === "Hindi" ? "font-hindi" : ""
            }`}
            style={{ height: "auto" }}
            onInput={(e) => {
              e.target.style.height = "auto";
              e.target.style.height = `${Math.min(e.target.scrollHeight, 128)}px`;
            }}
          />
          <div className="flex flex-col gap-1.5">
            <button
              onClick={() => handleSend()}
              disabled={loading || !input.trim()}
              className="btn-primary px-4 py-2.5 flex items-center gap-2 text-sm"
            >
              <span>{loading ? "⏳" : "📤"}</span>
              <span className="hidden sm:inline">
                {loading ? "Sending…" : "Send"}
              </span>
            </button>
            <button
              onClick={handleClear}
              disabled={loading}
              title="Clear chat"
              className="btn-secondary px-4 py-1.5 text-xs text-center"
            >
              🗑 Clear
            </button>
          </div>
        </div>
        <p className="text-xs text-gray-400 text-center mt-1.5">
          {language === "Hindi"
            ? "Shift+Enter से नई लाइन बनाएं"
            : "Shift+Enter for new line • Responses may take 10–30 seconds"}
        </p>
      </div>
    </div>
  );
}
