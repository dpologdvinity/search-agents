// Shared effects for the game pages: particle bursts, screen shake, floating
// score pops, and banners for big moments. One overlay canvas, drawn only
// while particles are alive. All effects are skipped under reduced motion,
// except banners, which carry information and appear without animation.

const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const COLORS = ['#00f5ff', '#ff00a0', '#9b00ff', '#00ff88', '#ffe600'];

let canvas, ctx, particles = [], raf = 0;

function overlay() {
  if (canvas) return;
  canvas = document.createElement('canvas');
  canvas.className = 'fx-layer';
  canvas.setAttribute('aria-hidden', 'true');
  document.body.appendChild(canvas);
  ctx = canvas.getContext('2d');
  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = innerWidth * dpr;
    canvas.height = innerHeight * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };
  resize();
  addEventListener('resize', resize);
}

function tick() {
  ctx.clearRect(0, 0, innerWidth, innerHeight);
  particles = particles.filter((p) => p.life > 0);
  for (const p of particles) {
    p.x += p.vx; p.y += p.vy; p.vy += p.g; p.vx *= 0.985; p.life -= 1;
    ctx.globalAlpha = Math.max(0, p.life / p.max);
    ctx.fillStyle = p.color;
    ctx.shadowColor = p.color; ctx.shadowBlur = 10;
    if (p.square) ctx.fillRect(p.x - p.r, p.y - p.r, p.r * 2, p.r * 2);
    else { ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill(); }
  }
  ctx.globalAlpha = 1; ctx.shadowBlur = 0;
  raf = particles.length ? requestAnimationFrame(tick) : 0;
}

function center(el) {
  const r = el.getBoundingClientRect();
  return [r.left + r.width / 2, r.top + r.height / 2];
}

/** Burst of particles from the centre of el (or [x, y]). */
export function burst(target, { count = 60, colors = COLORS, speed = 7, gravity = 0.12 } = {}) {
  if (REDUCED || !target) return;
  overlay();
  const [x, y] = Array.isArray(target) ? target : center(target);
  for (let i = 0; i < count; i++) {
    const a = Math.random() * Math.PI * 2, v = speed * (0.3 + Math.random());
    const life = 40 + Math.random() * 40;
    particles.push({ x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v - 2, g: gravity, r: 1.5 + Math.random() * 2.5,
                     color: colors[i % colors.length], life, max: life, square: Math.random() < 0.4 });
  }
  if (!raf) raf = requestAnimationFrame(tick);
}

/** Shake an element briefly. */
export function shake(el, strength = 'small') {
  if (REDUCED || !el) return;
  el.classList.remove('fx-shake', 'fx-shake-big');
  void el.offsetWidth; // restart the animation
  el.classList.add(strength === 'big' ? 'fx-shake-big' : 'fx-shake');
}

/** Floating text (e.g. "+64") that rises from el and fades. */
export function pop(target, text, color = '#ffe600') {
  if (REDUCED || !target) return;
  const [x, y] = Array.isArray(target) ? target : center(target);
  const el = document.createElement('div');
  el.className = 'fx-pop';
  el.textContent = text;
  el.style.left = `${x}px`; el.style.top = `${y}px`; el.style.color = color;
  document.body.appendChild(el);
  el.addEventListener('animationend', () => el.remove());
}

/** Big banner across the screen for a moment; `sub` is a smaller second line. */
export function banner(text, sub = '', color = '#00f5ff') {
  document.querySelectorAll('.fx-banner').forEach((b) => b.remove());
  const el = document.createElement('div');
  el.className = 'fx-banner';
  el.setAttribute('role', 'status');
  el.style.setProperty('--fx-c', color);
  el.innerHTML = `<span class="fx-banner-main"></span><span class="fx-banner-sub"></span>`;
  el.querySelector('.fx-banner-main').textContent = text;
  el.querySelector('.fx-banner-sub').textContent = sub;
  document.body.appendChild(el);
  setTimeout(() => el.classList.add('fx-out'), REDUCED ? 2600 : 2200);
  setTimeout(() => el.remove(), 3000);
}
