// ======================================================
// CONFIGURATION
// ======================================================
const CONFIG = {
  API_URL: window.location.port === "5000"
    ? "/predict"
    : "http://127.0.0.1:5000/predict",
  THRESHOLD: 0.5         // probability above this = high risk
};

// ======================================================
// VALIDATION RULES (min/max for numeric fields)
// ======================================================
const RULES = {
  age:      { min: 1,   max: 120, label: "Age" },
  trestbps: { min: 50,  max: 250, label: "Resting blood pressure" },
  chol:     { min: 50,  max: 700, label: "Cholesterol" },
  thalach:  { min: 50,  max: 250, label: "Max heart rate" },
  oldpeak:  { min: 0,   max: 10,  label: "ST depression" }
};

const REQUIRED_SELECTS = [
  "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
];

const form = document.getElementById("predictForm");
const submitBtn = document.getElementById("submitBtn");
const resetBtn = document.getElementById("resetBtn");
const resultCard = document.getElementById("resultCard");
const errorCard = document.getElementById("errorCard");

// ======================================================
// VALIDATION
// ======================================================
function clearErrors() {
  document.querySelectorAll(".error").forEach(el => (el.textContent = ""));
  document.querySelectorAll(".invalid").forEach(el => el.classList.remove("invalid"));
}

function showFieldError(id, message) {
  const input = document.getElementById(id);
  const errorEl = document.getElementById(id + "-error");
  if (input) input.classList.add("invalid");
  if (errorEl) errorEl.textContent = message;
}

function validateForm() {
  clearErrors();
  let valid = true;

  for (const [id, rule] of Object.entries(RULES)) {
    const raw = document.getElementById(id).value.trim();
    if (raw === "") {
      showFieldError(id, `${rule.label} is required.`);
      valid = false;
      continue;
    }
    const value = Number(raw);
    if (Number.isNaN(value)) {
      showFieldError(id, "Please enter a valid number.");
      valid = false;
    } else if (value < rule.min || value > rule.max) {
      showFieldError(id, `${rule.label} must be between ${rule.min} and ${rule.max}.`);
      valid = false;
    }
  }

  for (const id of REQUIRED_SELECTS) {
    if (document.getElementById(id).value === "") {
      showFieldError(id, "Please select an option.");
      valid = false;
    }
  }

  return valid;
}

// ======================================================
// DATA COLLECTION
// ======================================================
function getFormData() {
  const data = {};
  const ids = [...Object.keys(RULES), ...REQUIRED_SELECTS];

  ids.forEach(id => {
    data[id] = Number(document.getElementById(id).value);
  });

  return data;
}

// ======================================================
// PREDICTION (calls the Flask backend)
// ======================================================
async function getPrediction(data) {
  const response = await fetch(CONFIG.API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  });

  let body = {};
  try {
    body = await response.json();
  } catch {
    if (response.ok) {
      throw new Error("The backend returned an invalid response.");
    }
  }

  if (!response.ok) {
    throw new Error(
      body.error ||
      `Backend request failed with status ${response.status}. ` +
      "Make sure the Flask server is running on port 5000."
    );
  }

  if (!Number.isFinite(body.probability) || body.probability < 0 || body.probability > 1) {
    throw new Error("The backend returned an invalid prediction.");
  }

  // Expected: { "probability": 0.73, "prediction": 1 }
  return body;
}

// ======================================================
// DISPLAY
// ======================================================
function showResult(probability) {
  const percent = Math.round(probability * 100);
  const isHigh = probability >= CONFIG.THRESHOLD;

  const badge = document.getElementById("resultBadge");
  const bar = document.getElementById("resultBar");

  badge.textContent = isHigh ? "High Risk" : "Low Risk";
  badge.className = "badge " + (isHigh ? "high" : "low");

  bar.style.width = percent + "%";
  bar.style.background = isHigh ? "var(--danger)" : "var(--success)";

  document.getElementById("resultText").textContent = isHigh
    ? "The model indicates a higher likelihood of heart disease. Please consult a doctor."
    : "The model indicates a lower likelihood of heart disease. Keep up regular checkups.";

  document.getElementById("resultPercent").textContent =
    `Estimated probability: ${percent}%`;

  errorCard.classList.add("hidden");
  resultCard.classList.remove("hidden");
  resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

function showError(message) {
  document.getElementById("errorText").textContent = message;
  errorCard.classList.remove("hidden");
}

// ======================================================
// EVENT HANDLERS
// ======================================================
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resultCard.classList.add("hidden");
  errorCard.classList.add("hidden");

  if (!validateForm()) {
    const firstInvalid = form.querySelector(".invalid");
    if (firstInvalid) firstInvalid.focus();
    return;
  }

  const data = getFormData();

  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting...";

  try {
    const result = await getPrediction(data);
    showResult(result.probability);
  } catch (err) {
    console.error(err);
    const message = err instanceof TypeError
      ? "Could not connect to the Flask backend. Start it in a terminal with " +
        "\"python backend/app.py\" and keep it running on port 5000."
      : err.message || "Could not get a prediction. Please try again.";
    showError(message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict Risk";
  }
});

resetBtn.addEventListener("click", () => {
  clearErrors();
  resultCard.classList.add("hidden");
  errorCard.classList.add("hidden");
});