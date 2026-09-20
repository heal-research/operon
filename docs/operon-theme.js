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
    document.documentElement.classList.toggle("dark-mode", theme === "dark");
    document.documentElement.classList.toggle("light-mode", theme === "light");
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
})();
