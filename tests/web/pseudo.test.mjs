// The pseudocode colouring behind the explanation boxes: each token gets a type, and the tokens rebuild the source exactly.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { tokenizePseudo } from '../../web/js/pseudo.js';

// The non-plain tokens of a source, as [type, text] pairs, so a test can check one kind at a time.
function typed(source) {
  return tokenizePseudo(source).filter((t) => t.type !== 'text').map((t) => [t.type, t.text]);
}

test('tokens rebuild the source exactly, including newlines and symbols', () => {
  const source = 'function ALPHA-BETA(depth, α, β):\n    best ← −infinity   # minus is −\n    return best\n';
  assert.equal(tokenizePseudo(source).map((t) => t.text).join(''), source);
});

test('keywords are coloured and words that merely contain them are not', () => {
  assert.deepEqual(typed('for each move m in MOVES:\nreturn format'), [
    ['keyword', 'for'], ['keyword', 'each'], ['keyword', 'in'], ['proc', 'MOVES'], ['keyword', 'return'],
  ]);
});

test('capitalised procedure names, including hyphenated ones, are procedures', () => {
  assert.deepEqual(typed('x ← ALPHA-BETA(LEGAL-MOVES(p))'), [
    ['arrow', '←'], ['proc', 'ALPHA-BETA'], ['proc', 'LEGAL-MOVES'],
  ]);
});

test('a mixed-case word is not a procedure name', () => {
  assert.deepEqual(typed('Man Win'), []);
});

test('numbers, including decimals and the word infinity, are numbers', () => {
  assert.deepEqual(typed('best ← −infinity; a ← 2.5 + 40'), [
    ['arrow', '←'], ['minus', '−'], ['number', 'infinity'], ['arrow', '←'], ['number', '2.5'], ['number', '40'],
  ]);
});

test('builtins are coloured as builtins', () => {
  assert.deepEqual(typed('return max(best, sqrt(n))'), [
    ['keyword', 'return'], ['builtin', 'max'], ['builtin', 'sqrt'],
  ]);
});

test('a hash runs to the end of its line as a comment, and the next line is plain', () => {
  assert.deepEqual(typed('break   # prune here\nstop'), [
    ['keyword', 'break'], ['comment', '# prune here'], ['keyword', 'stop'],
  ]);
});

test('Greek letters and the arrow are coloured, and the minus sign has its own class', () => {
  assert.deepEqual(typed('α ← β − γ'), [
    ['greek', 'α'], ['arrow', '←'], ['greek', 'β'], ['minus', '−'], ['greek', 'γ'],
  ]);
});

test('an identifier with digits is one plain name, not a keyword and a number', () => {
  assert.deepEqual(typed('x1 ← score2'), [['arrow', '←']]);
});

test('the real minus sign is its own token, and the ASCII hyphen is plain text', () => {
  assert.deepEqual(typed('depth − 1 and -1'), [['minus', '−'], ['number', '1'], ['keyword', 'and'], ['number', '1']]);
  assert.deepEqual(typed('x ← −infinity # −1 lost'), [
    ['arrow', '←'], ['minus', '−'], ['number', 'infinity'], ['comment', '# '], ['minus', '−'], ['comment', '1 lost'],
  ]);
});

test('a minus sign inside a comment is a minus token that stays inside the comment', () => {
  const tokens = tokenizePseudo('# lost: −1, 0 draw');
  assert.deepEqual(tokens.map((t) => [t.type, t.text, t.within]), [
    ['comment', '# lost: ', undefined], ['minus', '−', 'comment'], ['comment', '1, 0 draw', undefined],
  ]);
});
