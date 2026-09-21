(() => {
  const maximumWidth = 900;

  const sizeDiagram = svg => {
    const { width, height } = svg.viewBox.baseVal;
    if (width <= 0 || height <= 0) return;
    const renderedWidth = Math.min(width, maximumWidth);
    svg.style.width = `${renderedWidth}px`;
    svg.style.height = `${Math.ceil(renderedWidth * height / width)}px`;
  };

  const sizeDiagrams = root => {
    root.querySelectorAll?.(".mermaid > svg").forEach(sizeDiagram);
  };

  document.addEventListener("DOMContentLoaded", () => {
    sizeDiagrams(document);
    new MutationObserver(records => {
      for (const record of records) {
        for (const node of record.addedNodes) {
          if (node.nodeType === Node.ELEMENT_NODE) sizeDiagrams(node);
        }
      }
    }).observe(document.body, { childList: true, subtree: true });
  });
})();
