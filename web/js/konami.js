// The Konami code as a pure matcher, kept apart from overdrive.js so node can test it without a DOM.
// konamiStep(progress, key) takes the number of keys matched so far and the next key, and returns the new
// progress plus whether the full code was just completed. A wrong key restarts the sequence, but a key that
// starts the code (ArrowUp) counts as its first key, so a stray press does not cost the whole attempt.

export const KONAMI = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 'b', 'a'];

/** Letters compare case-insensitively ('B' and 'b' are the same key); named keys are compared as given. */
export function normalizeKey(key) {
  return key.length === 1 ? key.toLowerCase() : key;
}

export function konamiStep(progress, key) {
  if (key === KONAMI[progress]) {
    const next = progress + 1;
    return next === KONAMI.length ? { progress: 0, done: true } : { progress: next, done: false };
  }
  return { progress: key === KONAMI[0] ? 1 : 0, done: false };
}
