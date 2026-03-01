const problemTypeFilter = document.getElementById("problemTypeFilter");
const answerStatusFilter = document.getElementById("answerStatusFilter");
const reviewerScopeFilter = document.getElementById("reviewerScopeFilter");
const confidenceFilter = document.getElementById("confidenceFilter");
const searchFilter = document.getElementById("searchFilter");
const visibleRows = document.getElementById("visibleRows");
const navToggle = document.getElementById("navToggle");
const rows = Array.from(document.querySelectorAll("#qaTable tbody tr[data-problem]"));
const NAV_STATE_KEY = "cc_nav_collapsed";

function normalizeProblem(value) {
  return String(value || "missing_info").trim().toLowerCase();
}

function parseExcludedFlag(value) {
  const normalized = String(value || "").trim().toLowerCase();
  return normalized === "1" || normalized === "true" || normalized === "yes";
}

function setNavCollapsed(collapsed) {
  document.body.classList.toggle("nav-collapsed", collapsed);
  if (navToggle) {
    navToggle.textContent = collapsed ? ">" : "||";
  }
}

function initSidebarToggle() {
  if (!navToggle) {
    return;
  }
  const saved = localStorage.getItem(NAV_STATE_KEY);
  setNavCollapsed(saved === "1");
  navToggle.addEventListener("click", () => {
    const next = !document.body.classList.contains("nav-collapsed");
    setNavCollapsed(next);
    localStorage.setItem(NAV_STATE_KEY, next ? "1" : "0");
  });
}

function applyFilters() {
  const selectedProblem = (problemTypeFilter?.value || "all").toLowerCase();
  const selectedAnswerStatus = (answerStatusFilter?.value || "all").toLowerCase();
  const selectedReviewerScope = (reviewerScopeFilter?.value || "all").toLowerCase();
  const maxConfidence = Number(confidenceFilter?.value || 1);
  const query = (searchFilter?.value || "").toLowerCase().trim();

  let shown = 0;
  rows.forEach((row) => {
    const rowProblem = normalizeProblem(row.dataset.problem);
    const rowConfidence = Number(row.dataset.confidence || 0);
    const rowQuestion = (row.dataset.question || "").toLowerCase();
    const rowExcluded = parseExcludedFlag(row.dataset.excluded || "0");

    const problemMatch = selectedProblem === "all" || rowProblem === selectedProblem;
    const statusMatch =
      selectedAnswerStatus === "all" ||
      (selectedAnswerStatus === "non_ok" && rowProblem !== "ok") ||
      (selectedAnswerStatus === "ok" && rowProblem === "ok");
    const reviewerScopeMatch =
      selectedReviewerScope === "all" ||
      (selectedReviewerScope === "included" && !rowExcluded) ||
      (selectedReviewerScope === "excluded" && rowExcluded);
    const confidenceMatch = rowConfidence <= maxConfidence;
    const queryMatch = !query || rowQuestion.includes(query);
    const isVisible = problemMatch && statusMatch && reviewerScopeMatch && confidenceMatch && queryMatch;
    row.style.display = isVisible ? "" : "none";
    if (isVisible) {
      shown += 1;
    }
  });
  if (visibleRows) {
    visibleRows.textContent = String(shown);
  }
}

problemTypeFilter?.addEventListener("change", applyFilters);
answerStatusFilter?.addEventListener("change", applyFilters);
reviewerScopeFilter?.addEventListener("change", applyFilters);
confidenceFilter?.addEventListener("input", applyFilters);
searchFilter?.addEventListener("input", applyFilters);
initSidebarToggle();
applyFilters();
