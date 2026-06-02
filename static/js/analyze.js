/* analyze.js — URL analysis form, API call, sample URL buttons */

document.addEventListener('DOMContentLoaded', function () {

  const urlInput    = document.getElementById('urlInput');
  const analyzeBtn  = document.getElementById('analyzeBtn');
  const clearBtn    = document.getElementById('urlClearBtn');
  const errorEl     = document.getElementById('analyzeError');

  if (!urlInput || !analyzeBtn) return;

  // ── Clear button ───────────────────────────────────────────────────
  urlInput.addEventListener('input', function () {
    if (clearBtn) clearBtn.style.display = this.value ? 'flex' : 'none';
  });
  if (clearBtn) {
    clearBtn.addEventListener('click', function () {
      urlInput.value = '';
      urlInput.focus();
      this.style.display = 'none';
    });
  }

  // Show clear button if prefilled
  if (urlInput.value && clearBtn) clearBtn.style.display = 'flex';

  // ── Sample URL buttons ─────────────────────────────────────────────
  document.querySelectorAll('.sample-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      const url = this.dataset.url;
      if (!url || !urlInput) return;
      urlInput.value = url;
      if (clearBtn) clearBtn.style.display = 'flex';
      urlInput.focus();
      // Smooth scroll to input on mobile
      urlInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
  });

  // ── Analyze form submission ────────────────────────────────────────
  analyzeBtn.addEventListener('click', runAnalysis);
  urlInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') runAnalysis();
  });

  function runAnalysis() {
    const url = (urlInput ? urlInput.value : '').trim();
    if (!url) {
      showError('Please enter a URL to analyze.');
      return;
    }

    setLoading(true);
    hideError();

    fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: url }),
    })
      .then(function (res) {
        return res.json().then(function (data) {
          return { ok: res.ok, data: data };
        });
      })
      .then(function ({ ok, data }) {
        setLoading(false);
        if (!ok || data.error) {
          showError(data.error || 'Analysis failed. Please check the URL and try again.');
          return;
        }
        // Store result and redirect
        sessionStorage.setItem('phishguard_result', JSON.stringify(data));
        window.location.href = '/result';
      })
      .catch(function (err) {
        setLoading(false);
        showError('Network error. Please check your connection and try again.');
        console.error('Analysis error:', err);
      });
  }

  function setLoading(on) {
    if (!analyzeBtn) return;
    const textEl    = analyzeBtn.querySelector('.analyze-btn-text');
    const spinnerEl = analyzeBtn.querySelector('.analyze-btn-spinner');
    analyzeBtn.disabled = on;
    if (textEl)    textEl.style.display    = on ? 'none'  : '';
    if (spinnerEl) spinnerEl.style.display = on ? 'flex'  : 'none';
    analyzeBtn.style.opacity = on ? '0.7' : '1';
  }

  function showError(msg) {
    if (!errorEl) return;
    errorEl.textContent = msg;
    errorEl.style.display = 'block';
  }

  function hideError() {
    if (!errorEl) return;
    errorEl.style.display = 'none';
  }

});
