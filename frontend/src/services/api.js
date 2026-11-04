import axios from "axios";

const API_BASE_URL =
  process.env.REACT_APP_API_URL ||
  (process.env.NODE_ENV === "production" ? "/api" : "http://localhost:8000/api");

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 minutes — LLM can be slow
  headers: {
    "Content-Type": "application/json",
  },
});

/**
 * Send a chat message to the KrishiRakshak backend agent.
 * @param {string} message - Farmer's question
 * @param {string} language - "English" or "Hindi"
 * @param {string} location - City name
 * @returns {Promise<{response: string, sources: string[], agent_steps: Array}>}
 */
export async function sendMessage(message, language = "English", location = "Delhi") {
  const response = await api.post("/chat", {
    message,
    language,
    location,
  });
  return response.data;
}

/**
 * Health check.
 * @returns {Promise<{status: string, service: string}>}
 */
export async function checkHealth() {
  const response = await api.get("/health");
  return response.data;
}

/**
 * Fetch supported crops from backend.
 * @returns {Promise<{crops: Array}>}
 */
export async function getCrops() {
  const response = await api.get("/crops");
  return response.data;
}

/**
 * Fetch supported markets from backend.
 * @returns {Promise<{markets: string[], date: string}>}
 */
export async function getMarkets() {
  const response = await api.get("/markets");
  return response.data;
}

/**
 * Send a plant image to the disease detection endpoint.
 * @param {File} imageFile - Image file object from <input type="file">
 * @param {number} topK - Number of top results (default 3)
 * @returns {Promise<DiseaseDetectionResponse>}
 */
export async function detectDisease(imageFile, topK = 3) {
  const formData = new FormData();
  formData.append("file", imageFile);

  const response = await api.post(`/detect-disease?top_k=${topK}`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 60000, // 60s — model inference can be slow on CPU
  });
  return response.data;
}

/**
 * Fetch supported disease classes from backend.
 * @returns {Promise<{total: number, diseases: Array, general_advice: object}>}
 */
export async function getDiseases() {
  const response = await api.get("/diseases");
  return response.data;
}

export default api;
