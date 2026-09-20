(() => {
  const storageKey = "operon-docs-theme";
  const windowNameKey = "operon-docs-theme";
  const themePattern = new RegExp(`(?:^|;)${windowNameKey}=(dark|light)(?:;|$)`);

  const storedTheme = () => {
    try {
      return localStorage.getItem(storageKey);
    } catch {
      return null;
    }
  };

  const windowTheme = () => window.name.match(themePattern)?.[1] ?? null;

  const applyTheme = theme => {
    if (theme !== "dark" && theme !== "light") return;
    const enabled = theme === "dark";
    try {
      DoxygenAwesomeDarkModeToggle.userPreference = enabled;
    } catch {
      // Browsers can deny storage to file:// previews. The native setter
      // persists its preference there, so fall back to its storage-free API.
      DoxygenAwesomeDarkModeToggle.enableDarkMode(enabled);
    }
    document.querySelector("doxygen-awesome-dark-mode-toggle")?.updateIcon();
  };

  const persistTheme = theme => {
    if (theme !== "dark" && theme !== "light") return;
    try {
      localStorage.setItem(storageKey, theme);
    } catch {
      // file:// previews can isolate localStorage per page; window.name survives
      // same-tab chapter navigation and preserves the selected preview theme.
    }
    const otherWindowState = window.name.replace(themePattern, ";").replace(/^;|;$/g, "");
    window.name = `${otherWindowState}${otherWindowState ? ";" : ""}${windowNameKey}=${theme}`;
  };

  const restoreTheme = () => applyTheme(storedTheme() ?? windowTheme());

  restoreTheme();
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") restoreTheme();
  });
  document.addEventListener("click", event => {
    const target = event.target instanceof Element ? event.target : event.target.parentElement;
    if (!target?.closest("doxygen-awesome-dark-mode-toggle")) return;
    queueMicrotask(() => persistTheme(document.documentElement.classList.contains("dark-mode") ? "dark" : "light"));
  });
  const scaleDiagram = frame => {
    const svg = frame.contentDocument?.documentElement;
    const dimensions = svg?.getAttribute("viewBox")?.trim().split(/\s+/).map(Number);
    if (!svg || !dimensions || dimensions.length !== 4 || dimensions.some(Number.isNaN)) return;
    const [, , width, height] = dimensions;
    if (width <= 0 || height <= 0) return;
    svg.setAttribute("width", "100%");
    svg.setAttribute("height", "100%");
    frame.style.width = "100%";
    frame.style.height = `${Math.ceil(frame.parentElement.clientWidth * height / width)}px`;
  };

  const scaleDiagrams = () => {
    for (const frame of document.querySelectorAll(".dotgraph iframe")) {
      frame.addEventListener("load", () => scaleDiagram(frame), { once: true });
      scaleDiagram(frame);
      new ResizeObserver(() => scaleDiagram(frame)).observe(frame.parentElement);
    }
  };

  document.addEventListener("DOMContentLoaded", scaleDiagrams);
})();
