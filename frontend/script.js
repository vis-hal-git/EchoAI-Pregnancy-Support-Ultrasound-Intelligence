/* ============================================================
   WombCare — Shared JavaScript
   ============================================================ */

'use strict';

// ── 1. SKELETON LOADER ─────────────────────────────────────
window.addEventListener('load', () => {
  const overlay = document.querySelector('.skeleton-overlay');
  if (overlay) {
    setTimeout(() => overlay.classList.add('hidden'), 400);
  }
});

// ── 2. NAVBAR ──────────────────────────────────────────────
const navbar     = document.querySelector('.navbar');
const hamburger  = document.querySelector('.nav-hamburger');
const navDrawer  = document.querySelector('.nav-drawer');

// Sticky shrink on scroll
window.addEventListener('scroll', () => {
  if (!navbar) return;
  if (window.scrollY > 60) {
    navbar.classList.add('scrolled');
  } else {
    navbar.classList.remove('scrolled');
  }
}, { passive: true });

// Hamburger toggle
if (hamburger && navDrawer) {
  hamburger.addEventListener('click', () => {
    hamburger.classList.toggle('open');
    navDrawer.classList.toggle('open');
    document.body.style.overflow = navDrawer.classList.contains('open') ? 'hidden' : '';
  });
  // Close on drawer link click
  navDrawer.querySelectorAll('a').forEach(a => {
    a.addEventListener('click', () => {
      hamburger.classList.remove('open');
      navDrawer.classList.remove('open');
      document.body.style.overflow = '';
    });
  });
}

// Active nav link based on current page
(function setActiveNavLink() {
  const page = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-links a, .nav-drawer a').forEach(a => {
    const href = a.getAttribute('href');
    if (href === page || (page === '' && href === 'index.html')) {
      a.classList.add('active');
    }
  });
})();

// ── 3. INTERSECTION OBSERVER — FADE UP ─────────────────────
const fadeObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        fadeObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.12 }
);

document.querySelectorAll('.fade-up').forEach(el => fadeObserver.observe(el));

// ── 4. TYPEWRITER EFFECT ───────────────────────────────────
function initTypewriter() {
  const el = document.querySelector('.hero-typewriter');
  if (!el) return;

  const phrases = el.dataset.phrases
    ? JSON.parse(el.dataset.phrases)
    : ['Fetal Ultrasound Intelligence', 'Pregnancy Decision Support', 'Multimodal Ultrasound AI'];

  let phraseIdx = 0;
  let charIdx   = 0;
  let deleting  = false;
  let paused    = false;

  const textSpan   = document.createElement('span');
  const cursorSpan = document.createElement('span');
  cursorSpan.className = 'cursor';
  el.appendChild(textSpan);
  el.appendChild(cursorSpan);

  function tick() {
    if (paused) return;
    const phrase = phrases[phraseIdx];

    if (!deleting) {
      textSpan.textContent = phrase.slice(0, charIdx + 1);
      charIdx++;
      if (charIdx === phrase.length) {
        paused = true;
        setTimeout(() => { paused = false; deleting = true; tick(); }, 2400);
        return;
      }
    } else {
      textSpan.textContent = phrase.slice(0, charIdx - 1);
      charIdx--;
      if (charIdx === 0) {
        deleting = false;
        phraseIdx = (phraseIdx + 1) % phrases.length;
      }
    }

    setTimeout(tick, deleting ? 50 : 85);
  }

  tick();
}
initTypewriter();

// ── 5. COUNTER ANIMATION ───────────────────────────────────
function animateCounter(el, target, suffix) {
  const duration = 2000;
  const startTime = performance.now();

  function update(now) {
    const elapsed  = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    // Ease out cubic
    const ease = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.floor(ease * target).toLocaleString() + (suffix || '');
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

const counterObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      const el = entry.target;
      const target = parseInt(el.dataset.target, 10);
      const suffix = el.dataset.suffix || '';
      animateCounter(el, target, suffix);
      counterObserver.unobserve(el);
    });
  },
  { threshold: 0.5 }
);

document.querySelectorAll('[data-target]').forEach(el => counterObserver.observe(el));

// ── 6. ACCORDION ───────────────────────────────────────────
document.querySelectorAll('.accordion-header').forEach(header => {
  header.addEventListener('click', () => {
    const item = header.closest('.accordion-item');
    const isOpen = item.classList.contains('open');

    // Close all
    document.querySelectorAll('.accordion-item.open').forEach(i => i.classList.remove('open'));

    // Toggle current
    if (!isOpen) item.classList.add('open');
  });
});

// ── 7. PIPELINE CARD KEYBOARD SUPPORT ─────────────────────
document.querySelectorAll('.pipeline-card').forEach(card => {
  card.setAttribute('tabindex', '0');
  card.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      card.classList.toggle('expanded');
    }
  });
});

// ── 8. SMOOTH SCROLL FOR ANCHOR LINKS ─────────────────────
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      e.preventDefault();
      const offset = 80; // navbar height
      const top = target.getBoundingClientRect().top + window.scrollY - offset;
      window.scrollTo({ top, behavior: 'smooth' });
    }
  });
});

// ── 9. APPLY FADE-UP TO CARDS (lazy) ──────────────────────
// Cards that appear mid-content get staggered delays
document.querySelectorAll(
  '.feature-card, .how-step, .tech-card, .kb-card, .standard-card, ' +
  '.threshold-card, .timeline-content'
).forEach((el, idx) => {
  if (!el.classList.contains('fade-up')) {
    el.classList.add('fade-up');
    el.classList.add(`delay-${(idx % 5) + 1}`);
    fadeObserver.observe(el);
  }
});

// ── 10. LIVE CLINICAL SCAN ANALYZER INTEGRATION ───────────
(function initAnalyzer() {
  const dropZone       = document.getElementById('drop-zone');
  const fileInput      = document.getElementById('file-input');
  const dropPrompt     = document.getElementById('drop-prompt');
  const previewWrap    = document.getElementById('preview-wrap');
  const previewImg     = document.getElementById('preview-img');
  const previewVid     = document.getElementById('preview-vid');
  const fileMeta       = document.getElementById('file-meta');
  const changeFileBtn  = document.getElementById('change-file-btn');
    const analyzeBtn     = document.getElementById('analyze-btn');
  const resetBtn       = document.getElementById('reset-btn');
  const progressBar    = document.getElementById('analysis-progress');
  const progressText   = document.getElementById('progress-text');
  const statusPill     = document.getElementById('backend-status-pill');
  const statusText     = document.getElementById('backend-status-text');

  const reportEmpty    = document.getElementById('report-empty');
  const reportContent  = document.getElementById('report-content');
  const triageBanner   = document.getElementById('triage-banner');
  const triageTitle    = document.getElementById('triage-title');
  const triageSub      = document.getElementById('triage-sub');
  const reportMeta     = document.getElementById('report-meta');
  const reportBody     = document.getElementById('report-body');
  const downloadBtn    = document.getElementById('download-report-btn');
  const citationsSec   = document.getElementById('citations-section');
  const citationsList  = document.getElementById('citations-list');
  const citationCount  = document.getElementById('citation-count');
      
  if (!dropZone || !analyzeBtn) return; // Not on page with analyzer

  let currentFile = null;
  let currentSessionId = null;
  let currentReportMarkdown = '';

  // Check Backend Health
  async function checkHealth() {
    try {
      const t0 = performance.now();
      const res = await fetch('/health');
      const latency = Math.round(performance.now() - t0);
      if (res.ok) {
        statusPill.className = 'status-pill status-online';
        statusText.textContent = `AI Backend Online · ${latency}ms`;
        return;
      }
    } catch (e) {
      // Backend error
    }
    statusPill.className = 'status-pill status-offline';
    statusText.textContent = 'AI Backend Offline';
  }
  checkHealth();

  // File selection & drop handlers
  dropZone.addEventListener('click', () => {
    if (!currentFile) fileInput.click();
  });

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files[0]) {
      handleFile(fileInput.files[0]);
    }
  });

  changeFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    currentFile = null;
    fileInput.value = '';
    previewWrap.style.display = 'none';
    dropPrompt.style.display = 'block';
    fileInput.click();
  });

  function handleFile(file) {
    currentFile = file;
    dropPrompt.style.display = 'none';
    previewWrap.style.display = 'block';

    const isVideo = file.type.startsWith('video') || file.name.match(/\.(mp4|avi|mov|mkv)$/i);
    const sizeKb = (file.size / 1024).toFixed(1);
    const sizeStr = sizeKb > 1024 ? (sizeKb / 1024).toFixed(2) + ' MB' : sizeKb + ' KB';
    fileMeta.textContent = `🖼️ ${file.name} (${sizeStr}) · ${isVideo ? 'Cine-loop video' : 'Ultrasound scan'}`;

    if (isVideo) {
      previewImg.style.display = 'none';
      previewVid.style.display = 'block';
      previewVid.src = URL.createObjectURL(file);
    } else {
      previewVid.style.display = 'none';
      previewImg.style.display = 'block';
      previewImg.src = URL.createObjectURL(file);
    }
  }

  // Reset button
  resetBtn.addEventListener('click', () => {
    currentFile = null;
    currentSessionId = null;
    currentReportMarkdown = '';
    fileInput.value = '';
    previewWrap.style.display = 'none';
    dropPrompt.style.display = 'block';
    reportContent.style.display = 'none';
    reportEmpty.style.display = 'block';
          });

  // Analyze button
  analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) {
      alert('⚠️ Please select or drop an ultrasound scan first.');
      return;
    }

    analyzeBtn.disabled = true;
    progressBar.style.display = 'block';

    const steps = [
      'Consulting clinical knowledge base (rag-only)...',
      'Ingesting scan & processing keyframes...',
      'Executing multimodal AI reasoning...',
      'Validating ISUOG planes & biometry...',
      'Structuring final assessment report...'
    ];
    let stepIdx = 0;
    progressText.textContent = steps[0];
    const timer = setInterval(() => {
      stepIdx = (stepIdx + 1) % steps.length;
      progressText.textContent = steps[stepIdx];
    }, 2800);

    const formData = new FormData();
    formData.append('file', currentFile);
    const query = 'Perform a comprehensive ultrasound evaluation: determine status (gravid vs nongravid), verify standard planes, evaluate biometry, and cite clinical guidelines.';
    formData.append('query', query);
    formData.append('k', '8');

    try {
      const res = await fetch('/analyze', {
        method: 'POST',
        body: formData
      });

      clearInterval(timer);
      analyzeBtn.disabled = false;
      progressBar.style.display = 'none';

      if (!res.ok) {
        const err = await res.text();
        alert(`Analysis error (${res.status}): ${err}`);
        return;
      }

      const data = await res.json();
      currentSessionId = data.session_id;
      currentReportMarkdown = data.answer;

      displayReport(data);
    } catch (err) {
      clearInterval(timer);
      analyzeBtn.disabled = false;
      progressBar.style.display = 'none';
      alert('Could not connect to AI backend: ' + err.message);
    }
  });

  // Render Report
  function displayReport(data) {
    reportEmpty.style.display = 'none';
    reportContent.style.display = 'block';

    const answer = data.answer || '';
    reportMeta.textContent = `Session: ${data.session_id.slice(0, 8)} · ${data.frame_count || 1} frame(s) processed`;

    // Triage Classification Banner
    if (answer.includes('PREGNANCY DETECTED') || answer.includes('GRAVID')) {
      triageBanner.className = 'triage-banner banner-gravid';
      triageTitle.textContent = '🤰 PREGNANCY DETECTED (GRAVID)';
      triageSub.textContent = 'Obstetric anatomical structures identified and grounded in ISUOG screening guidelines.';
    } else if (answer.includes('NON-PREGNANT') || answer.includes('NON-GRAVID')) {
      triageBanner.className = 'triage-banner banner-nongravid';
      triageTitle.textContent = '🩺 NON-PREGNANT (NON-GRAVID)';
      triageSub.textContent = 'Normal non-gravid pelvic morphology identified without intrauterine gestational sac.';
    } else {
      triageBanner.className = 'triage-banner banner-indeterminate';
      triageTitle.textContent = '⚠️ INDETERMINATE / PUL';
      triageSub.textContent = 'Equivocal findings. Clinical correlation with serum β-hCG and repeat TVS advised.';
    }

    // Render markdown to HTML
    reportBody.innerHTML = parseMarkdown(answer);

    // Render Citations
    if (data.citations && data.citations.length > 0) {
      citationsSec.style.display = 'block';
      citationCount.textContent = data.citations.length;
      citationsList.innerHTML = data.citations.map(c => `
        <div class="citation-item">
          <div class="citation-item-header">📖 ${escapeHtml(c.label || 'Clinical Record')}</div>
          <div style="color:var(--ink-soft); font-size:0.83rem; white-space:pre-wrap;">${escapeHtml(c.text || '')}</div>
        </div>
      `).join('');
    } else {
      citationsSec.style.display = 'none';
    }

    

    // Scroll to report smoothly
    document.getElementById('report-card').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // Download Report
  downloadBtn.addEventListener('click', () => {
    if (!currentReportMarkdown) return;
    const blob = new Blob([currentReportMarkdown], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `wombcare_assessment_${currentSessionId ? currentSessionId.slice(0, 8) : 'report'}.md`;
    a.click();
    URL.revokeObjectURL(url);
  });

  

  // Markdown Parser
  function parseMarkdown(md) {
    if (!md) return '';
    let html = escapeHtml(md);

    // Headers
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

    // Bold & Italic
    html = html.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/gim, '<em>$1</em>');

    // Blockquotes
    html = html.replace(/^\> (.*$)/gim, '<blockquote>$1</blockquote>');

    // Tables
    const lines = html.split('\n');
    let inTable = false;
    let tableHtml = '';
    const newLines = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (line.startsWith('|') && line.endsWith('|')) {
        if (!inTable) {
          inTable = true;
          tableHtml = '<table>';
        }
        if (line.includes('---')) continue; // Skip divider
        const cells = line.split('|').slice(1, -1).map(c => c.trim());
        const isHeader = !tableHtml.includes('<tbody>');
        if (isHeader) {
          tableHtml += '<thead><tr>' + cells.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
        } else {
          tableHtml += '<tr>' + cells.map(c => `<td>${c}</td>`).join('') + '</tr>';
        }
      } else {
        if (inTable) {
          inTable = false;
          tableHtml += '</tbody></table>';
          newLines.push(tableHtml);
          tableHtml = '';
        }
        newLines.push(lines[i]);
      }
    }
    if (inTable) {
      tableHtml += '</tbody></table>';
      newLines.push(tableHtml);
    }
    html = newLines.join('\n');

    // Lists
    html = html.replace(/^\s*-\s+(.*$)/gim, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/gim, '<ul>$1</ul>');
    html = html.replace(/<\/ul>\s*<ul>/gim, '');

    // Paragraphs
    html = html.split('\n\n').map(p => {
      p = p.trim();
      if (p.startsWith('<h') || p.startsWith('<table') || p.startsWith('<ul') || p.startsWith('<block')) {
        return p;
      }
      return p ? `<p>${p.replace(/\n/g, '<br>')}</p>` : '';
    }).join('\n');

    return html;
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
})();
