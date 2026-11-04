import React, { useCallback, useRef, useState } from "react";
import { detectDisease } from "../services/api";

// ── Severity config ────────────────────────────────────────────────────────────
const SEVERITY_CONFIG = {
  critical: { color: "bg-red-100 border-red-400 text-red-800", badge: "bg-red-500", icon: "🔴", label: "Critical" },
  high:     { color: "bg-orange-100 border-orange-400 text-orange-800", badge: "bg-orange-500", icon: "🟠", label: "High" },
  medium:   { color: "bg-yellow-100 border-yellow-400 text-yellow-800", badge: "bg-yellow-500", icon: "🟡", label: "Medium" },
  none:     { color: "bg-green-100 border-green-400 text-green-800", badge: "bg-green-500", icon: "🟢", label: "Healthy" },
  unknown:  { color: "bg-gray-100 border-gray-400 text-gray-800", badge: "bg-gray-500", icon: "⚪", label: "Unknown" },
};

// ── Treatment Section ─────────────────────────────────────────────────────────
function TreatmentCard({ treatment, generalAdvice }) {
  const [showGeneral, setShowGeneral] = useState(false);

  if (!treatment) return null;

  const sev = SEVERITY_CONFIG[treatment.severity] || SEVERITY_CONFIG.unknown;

  return (
    <div className="mt-4 space-y-3">
      {/* Disease header */}
      <div className={`rounded-xl border-2 p-4 ${sev.color}`}>
        <div className="flex items-start justify-between gap-2">
          <div>
            <h3 className="font-bold text-base">{treatment.common_name}</h3>
            {treatment.hindi_name && (
              <p className="text-sm opacity-80">{treatment.hindi_name}</p>
            )}
          </div>
          <span className={`text-xs font-bold px-2 py-1 rounded-full text-white ${sev.badge} shrink-0`}>
            {sev.icon} {sev.label}
          </span>
        </div>
        <p className="text-xs mt-1 opacity-75">Crop: {treatment.crop}</p>
      </div>

      {/* Treatment grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <InfoBlock title="🔬 Cause" content={treatment.cause} />
        <InfoBlock title="🌡️ Favorable Conditions" content={treatment.favorable_conditions} />
        <InfoBlock title="🌿 Symptoms" content={treatment.symptoms} className="sm:col-span-2" />
        <InfoBlock title="🍃 Organic Treatment" content={treatment.organic_treatment} accent="green" />
        <InfoBlock title="💊 Chemical Treatment" content={treatment.chemical_treatment} accent="blue" />
        <InfoBlock title="🛡️ Prevention" content={treatment.prevention} className="sm:col-span-2" accent="purple" />
        <InfoBlock title="💰 Economic Impact" content={treatment.economic_impact} className="sm:col-span-2" />
      </div>

      {/* General advice toggle */}
      {generalAdvice && Object.keys(generalAdvice).length > 0 && (
        <div>
          <button
            onClick={() => setShowGeneral(!showGeneral)}
            className="text-sm text-farm-600 hover:text-farm-800 font-medium flex items-center gap-1 transition-colors"
          >
            {showGeneral ? "▲" : "▼"} General Farming Advice
          </button>
          {showGeneral && (
            <div className="mt-2 bg-farm-50 border border-farm-200 rounded-lg p-3 text-xs text-gray-700 space-y-2">
              {Object.entries(generalAdvice).map(([key, val]) => (
                <p key={key}>
                  <span className="font-semibold capitalize">{key.replace(/_/g, " ")}: </span>
                  {val}
                </p>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function InfoBlock({ title, content, className = "", accent }) {
  const accentClass = {
    green: "border-l-4 border-green-400",
    blue: "border-l-4 border-blue-400",
    purple: "border-l-4 border-purple-400",
  }[accent] || "";

  return (
    <div className={`bg-white rounded-lg border border-gray-200 p-3 ${accentClass} ${className}`}>
      <p className="text-xs font-semibold text-gray-500 mb-1">{title}</p>
      <p className="text-sm text-gray-800 leading-relaxed">{content || "N/A"}</p>
    </div>
  );
}

// ── Detection Badge ────────────────────────────────────────────────────────────
function DetectionBadge({ detection, index, isSelected, onClick }) {
  const sev = SEVERITY_CONFIG[detection.severity] || SEVERITY_CONFIG.unknown;
  const confPct = (detection.confidence * 100).toFixed(1);

  return (
    <button
      onClick={onClick}
      className={`w-full text-left rounded-lg border-2 p-3 transition-all duration-200 ${
        isSelected
          ? "border-farm-500 bg-farm-50 shadow-sm"
          : "border-gray-200 bg-white hover:border-farm-300 hover:bg-farm-50"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-base shrink-0">{index === 0 ? "🎯" : `#${index + 1}`}</span>
          <div className="min-w-0">
            <p className="font-semibold text-sm text-gray-900 truncate">{detection.display_name}</p>
            {detection.hindi_name && (
              <p className="text-xs text-gray-500 truncate">{detection.hindi_name}</p>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {/* Confidence bar */}
          <div className="hidden sm:flex flex-col items-end gap-0.5">
            <span className="text-xs font-bold text-gray-700">{confPct}%</span>
            <div className="w-16 h-1.5 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-farm-500 rounded-full transition-all duration-500"
                style={{ width: `${confPct}%` }}
              />
            </div>
          </div>
          <span className={`text-xs font-bold px-2 py-0.5 rounded-full text-white ${sev.badge}`}>
            {sev.icon}
          </span>
        </div>
      </div>
    </button>
  );
}

// ── Main DiseaseDetector Component ────────────────────────────────────────────
export default function DiseaseDetector({ language }) {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const isHindi = language === "Hindi";

  const TEXT = {
    title: isHindi ? "🔍 रोग पहचान" : "🔍 Disease Detection",
    subtitle: isHindi
      ? "पौधे की तस्वीर अपलोड करें — AI रोग पहचानेगा"
      : "Upload a plant / leaf photo — AI detects diseases instantly",
    dropzone: isHindi ? "यहाँ तस्वीर छोड़ें या क्लिक करें" : "Drop an image here or click to browse",
    formats: isHindi ? "JPEG, PNG, WebP • अधिकतम 20MB" : "JPEG, PNG, WebP • max 20MB",
    analyze: isHindi ? "रोग पहचानें" : "Detect Disease",
    analyzing: isHindi ? "जाँच हो रही है..." : "Analyzing...",
    clear: isHindi ? "साफ करें" : "Clear",
    results: isHindi ? "पहचान परिणाम" : "Detection Results",
    noDisease: isHindi
      ? "✅ कोई रोग नहीं मिला — पौधा स्वस्थ दिखता है!"
      : "✅ No disease detected — plant looks healthy!",
    errorHint: isHindi
      ? "कृपया स्पष्ट तस्वीर अपलोड करें।"
      : "Please upload a clear, well-lit photo of the affected leaf or plant.",
    modelInfo: isHindi ? "मॉडल:" : "Model:",
    inferenceTime: isHindi ? "समय:" : "Time:",
    topDetections: isHindi ? "शीर्ष पहचान" : "Top Detections",
    treatment: isHindi ? "उपचार विवरण" : "Treatment Details",
  };

  // ── File handling ────────────────────────────────────────────────────────
  const handleFile = useCallback((file) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("Please upload an image file (JPEG, PNG, or WebP).");
      return;
    }
    setImageFile(file);
    setResult(null);
    setError(null);
    setSelectedIndex(0);
    const reader = new FileReader();
    reader.onload = (e) => setImagePreview(e.target.result);
    reader.readAsDataURL(file);
  }, []);

  const handleInputChange = (e) => handleFile(e.target.files[0]);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragOver(false);
      handleFile(e.dataTransfer.files[0]);
    },
    [handleFile]
  );

  const handleDragOver = (e) => { e.preventDefault(); setDragOver(true); };
  const handleDragLeave = () => setDragOver(false);

  const handleClear = () => {
    setImageFile(null);
    setImagePreview(null);
    setResult(null);
    setError(null);
    setSelectedIndex(0);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // ── Detection ────────────────────────────────────────────────────────────
  const handleDetect = async () => {
    if (!imageFile) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await detectDisease(imageFile, 3);
      setResult(data);
      setSelectedIndex(0);
    } catch (err) {
      const msg =
        err.response?.data?.detail ||
        err.message ||
        "Detection failed. Please try again.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  // ── Selected detection + treatment ──────────────────────────────────────
  const selectedDetection = result?.detections?.[selectedIndex] || null;
  const treatmentForSelected =
    selectedIndex === 0
      ? result?.treatment
      : null; // Only primary has pre-fetched treatment; others shown from detections

  // ── Render ───────────────────────────────────────────────────────────────
  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="bg-white rounded-2xl border border-gray-200 p-5 shadow-sm">
        <h2 className="text-xl font-bold text-farm-800">{TEXT.title}</h2>
        <p className="text-sm text-gray-500 mt-1">{TEXT.subtitle}</p>
      </div>

      {/* Upload zone */}
      <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
        <div
          className={`relative border-2 border-dashed rounded-xl m-4 transition-all duration-200 cursor-pointer ${
            dragOver
              ? "border-farm-500 bg-farm-50 scale-[0.99]"
              : imagePreview
              ? "border-farm-300 bg-farm-50"
              : "border-gray-300 hover:border-farm-400 hover:bg-gray-50"
          }`}
          onClick={() => fileInputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="hidden"
            onChange={handleInputChange}
          />

          {imagePreview ? (
            /* Image preview */
            <div className="flex flex-col sm:flex-row items-center gap-4 p-4">
              <div className="relative shrink-0">
                <img
                  src={imagePreview}
                  alt="Plant to analyze"
                  className="w-40 h-40 object-cover rounded-xl border-2 border-farm-300 shadow"
                />
                <div className="absolute -top-2 -right-2 bg-farm-500 text-white text-xs font-bold rounded-full w-6 h-6 flex items-center justify-center shadow">
                  ✓
                </div>
              </div>
              <div className="text-center sm:text-left">
                <p className="font-semibold text-gray-800 truncate max-w-xs">{imageFile?.name}</p>
                <p className="text-sm text-gray-500 mt-1">
                  {(imageFile?.size / 1024).toFixed(1)} KB
                </p>
                <p className="text-xs text-farm-600 mt-2">Click to change image</p>
              </div>
            </div>
          ) : (
            /* Empty dropzone */
            <div className="py-12 px-6 text-center">
              <div className="text-5xl mb-3 select-none">🌿</div>
              <p className="text-gray-600 font-medium">{TEXT.dropzone}</p>
              <p className="text-xs text-gray-400 mt-1">{TEXT.formats}</p>
              <div className="mt-4 flex flex-wrap justify-center gap-2 text-xs text-gray-500">
                {["🍅 Tomato", "🌾 Wheat", "🍚 Rice", "🍎 Apple", "🥔 Potato", "🌽 Corn", "🍇 Grape"].map((crop) => (
                  <span key={crop} className="bg-gray-100 px-2 py-0.5 rounded-full">{crop}</span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Action buttons */}
        <div className="flex gap-3 px-4 pb-4">
          <button
            onClick={handleDetect}
            disabled={!imageFile || loading}
            className={`flex-1 py-3 px-4 rounded-xl font-semibold text-sm transition-all duration-200 flex items-center justify-center gap-2 ${
              !imageFile || loading
                ? "bg-gray-200 text-gray-400 cursor-not-allowed"
                : "bg-farm-600 hover:bg-farm-700 text-white shadow-sm hover:shadow-md active:scale-[0.98]"
            }`}
          >
            {loading ? (
              <>
                <span className="inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                {TEXT.analyzing}
              </>
            ) : (
              <>
                <span>🔬</span>
                {TEXT.analyze}
              </>
            )}
          </button>

          {imageFile && (
            <button
              onClick={handleClear}
              className="py-3 px-4 rounded-xl font-semibold text-sm bg-gray-100 hover:bg-gray-200 text-gray-600 transition-colors"
            >
              {TEXT.clear}
            </button>
          )}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4">
          <div className="flex items-start gap-2">
            <span className="text-red-500 text-lg shrink-0">⚠️</span>
            <div>
              <p className="font-semibold text-red-700 text-sm">Detection Error</p>
              <p className="text-red-600 text-sm mt-0.5">{error}</p>
              <p className="text-red-500 text-xs mt-1">{TEXT.errorHint}</p>
            </div>
          </div>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
          {/* Result header */}
          <div className="bg-farm-700 text-white px-5 py-3 flex items-center justify-between">
            <h3 className="font-bold">{TEXT.results}</h3>
            <div className="text-xs text-farm-200 flex items-center gap-3">
              <span>{TEXT.inferenceTime} {result.inference_time_ms?.toFixed(0)}ms</span>
              <span className="hidden sm:inline">{TEXT.modelInfo} {result.model}</span>
            </div>
          </div>

          <div className="p-4">
            {/* No detections */}
            {(!result.detections || result.detections.length === 0) && (
              <div className="text-center py-8">
                <div className="text-5xl mb-2">✅</div>
                <p className="font-semibold text-green-700 text-lg">{TEXT.noDisease}</p>
                <p className="text-sm text-gray-500 mt-1">{result.message}</p>
              </div>
            )}

            {/* Detections found */}
            {result.detections && result.detections.length > 0 && (
              <div className="space-y-4">
                {/* Top detections list */}
                <div>
                  <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
                    {TEXT.topDetections}
                  </p>
                  <div className="space-y-2">
                    {result.detections.map((det, i) => (
                      <DetectionBadge
                        key={i}
                        detection={det}
                        index={i}
                        isSelected={selectedIndex === i}
                        onClick={() => setSelectedIndex(i)}
                      />
                    ))}
                  </div>
                </div>

                {/* Summary message */}
                {result.message && (
                  <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
                    <p className="text-sm text-amber-800">{result.message}</p>
                  </div>
                )}

                {/* Treatment for selected detection */}
                {selectedIndex === 0 && result.treatment && (
                  <div>
                    <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
                      {TEXT.treatment}
                    </p>
                    <TreatmentCard
                      treatment={result.treatment}
                      generalAdvice={result.general_advice}
                    />
                  </div>
                )}

                {/* For non-primary selections, show basic info from detection */}
                {selectedIndex > 0 && selectedDetection && (
                  <div className="bg-gray-50 border border-gray-200 rounded-xl p-4">
                    <p className="font-semibold text-gray-800">{selectedDetection.display_name}</p>
                    {selectedDetection.hindi_name && (
                      <p className="text-sm text-gray-500">{selectedDetection.hindi_name}</p>
                    )}
                    <p className="text-sm text-gray-600 mt-2">
                      Confidence: <strong>{(selectedDetection.confidence * 100).toFixed(1)}%</strong> •
                      Severity: <strong>{selectedDetection.severity}</strong>
                    </p>
                    <p className="text-xs text-gray-400 mt-2">
                      Select this result and re-scan or ask the AI chat about "{selectedDetection.display_name}" for full treatment advice.
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Usage tips */}
      {!result && !loading && (
        <div className="bg-farm-50 border border-farm-200 rounded-xl p-4">
          <p className="text-sm font-semibold text-farm-800 mb-2">📸 Tips for best results</p>
          <ul className="text-xs text-farm-700 space-y-1">
            <li>✓ Photograph a single affected leaf clearly against a plain background</li>
            <li>✓ Ensure good lighting — avoid shadows and blur</li>
            <li>✓ Include both healthy and affected areas in the frame</li>
            <li>✓ Supported crops: Tomato, Wheat, Rice, Apple, Potato, Corn, Grape, Cotton</li>
          </ul>
        </div>
      )}
    </div>
  );
}
