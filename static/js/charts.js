/* charts.js — Plotly.js SHAP visualizations for PhishGuard */

/* ── Design tokens (must match style.css) ───────────────────────────── */
const COLORS = {
  phish:   '#DC2626',
  legit:   '#16A34A',
  blue:    '#2563EB',
  teal:    '#0D9488',
  neutral: '#9CA3AF',
  text:    '#111827',
  muted:   '#6B7280',
  grid:    'rgba(0,0,0,0.06)',
  phishBg: 'rgba(220,38,38,0.12)',
  legitBg: 'rgba(22,163,74,0.12)',
};

const LAYOUT_BASE = {
  paper_bgcolor: 'transparent',
  plot_bgcolor:  'transparent',
  font:  { family: 'DM Sans, system-ui, sans-serif', color: COLORS.text, size: 12 },
  margin: { t: 20, r: 20, b: 60, l: 20 },
  xaxis: { gridcolor: COLORS.grid, zeroline: false },
  yaxis: { gridcolor: COLORS.grid, zeroline: false },
};

const CONFIG = {
  displayModeBar: false,
  responsive: true,
};

/* ── Feature name formatter ──────────────────────────────────────────── */
function fmtFeature(name) {
  const map = {
    having_ip_address:     'IP Address in URL',
    url_length:            'URL Length',
    shortening_service:    'URL Shortener',
    having_at_symbol:      '@ Symbol',
    double_slash_redirect: 'Double Slash',
    prefix_suffix_hyphen:  'Hyphen in Domain',
    subdomain_depth:       'Subdomain Depth',
    https_present:         'HTTPS Protocol',
    https_in_domain:       'HTTPS in Domain',
    non_standard_port:     'Non-std Port',
    submitting_to_email:   'Email Submit',
    abnormal_url:          'Abnormal URL',
    url_depth:             'URL Depth',
    suspicious_tld:        'Suspicious TLD',
    dns_record:            'DNS Record',
    domain_age:            'Domain Age',
  };
  return map[name] || name;
}

/* ═══════════════════════════════════════════════════════════════════════
   SHAP Waterfall Chart
   Shows cumulative contribution of each feature to the final prediction.
═══════════════════════════════════════════════════════════════════════ */
function renderShapWaterfall(data) {
  const el = document.getElementById('shapWaterfall');
  if (!el || typeof Plotly === 'undefined') return;

  const names = data.feature_names.map(fmtFeature);
  const vals  = data.shap_values;   // array of floats, same order

  // Sort by |SHAP| descending for waterfall readability
  const paired = names.map((n, i) => ({ name: n, val: vals[i] }));
  paired.sort((a, b) => Math.abs(b.val) - Math.abs(a.val));

  // Build waterfall: show top 10 for readability
  const top = paired.slice(0, 10);

  // Running total for waterfall positioning
  let running = 0;
  const measures  = [];
  const x         = [];
  const y         = [];
  const colors     = [];
  const textLabels = [];

  top.forEach(function (item) {
    measures.push('relative');
    x.push(item.name);
    y.push(item.val);
    colors.push(item.val > 0 ? COLORS.phish : COLORS.legit);
    textLabels.push((item.val > 0 ? '+' : '') + item.val.toFixed(3));
    running += item.val;
  });

  // Final total bar
  measures.push('total');
  x.push('Final Score');
  y.push(running);
  colors.push(running > 0 ? COLORS.phish : COLORS.legit);
  textLabels.push((running > 0 ? '+' : '') + running.toFixed(3));

  const trace = {
    type: 'waterfall',
    orientation: 'v',
    x: x,
    y: y,
    measure: measures,
    text: textLabels,
    textposition: 'outside',
    connector: {
      line: { color: COLORS.grid, width: 1.5 },
    },
    increasing: { marker: { color: COLORS.phish } },
    decreasing: { marker: { color: COLORS.legit } },
    totals:     { marker: { color: COLORS.blue  } },
  };

  const layout = Object.assign({}, LAYOUT_BASE, {
    height: 400,
    margin: { t: 30, r: 20, b: 100, l: 20 },
    xaxis: Object.assign({}, LAYOUT_BASE.xaxis, {
      tickangle: -35,
      tickfont: { size: 11 },
    }),
    yaxis: Object.assign({}, LAYOUT_BASE.yaxis, {
      title: { text: 'SHAP value (phishing direction)', font: { size: 11, color: COLORS.muted } },
      zeroline: true,
      zerolinecolor: COLORS.grid,
      zerolinewidth: 2,
    }),
    annotations: [{
      x: x[x.length - 1],
      y: running,
      text: running > 0 ? 'Phishing ↑' : 'Legitimate ↓',
      showarrow: false,
      font: {
        size: 11,
        color: running > 0 ? COLORS.phish : COLORS.legit,
        family: 'DM Sans, sans-serif',
      },
      yshift: running > 0 ? 20 : -20,
    }],
  });

  Plotly.newPlot(el, [trace], layout, CONFIG);
}

/* ═══════════════════════════════════════════════════════════════════════
   SHAP Horizontal Bar Chart (local — this URL)
═══════════════════════════════════════════════════════════════════════ */
function renderShapBar(data) {
  const el = document.getElementById('shapBar');
  if (!el || typeof Plotly === 'undefined') return;

  const paired = data.feature_names.map((n, i) => ({
    name: fmtFeature(n),
    val:  data.shap_values[i],
  }));
  paired.sort((a, b) => a.val - b.val);

  const trace = {
    type: 'bar',
    orientation: 'h',
    x: paired.map(p => p.val),
    y: paired.map(p => p.name),
    marker: {
      color: paired.map(p => p.val > 0 ? COLORS.phishBg : COLORS.legitBg),
      line: {
        color: paired.map(p => p.val > 0 ? COLORS.phish : COLORS.legit),
        width: 1.5,
      },
    },
    text: paired.map(p => (p.val > 0 ? '+' : '') + p.val.toFixed(3)),
    textposition: 'outside',
    textfont: {
      size: 10,
      family: 'DM Mono, monospace',
      color: paired.map(p => p.val > 0 ? COLORS.phish : COLORS.legit),
    },
  };

  const layout = Object.assign({}, LAYOUT_BASE, {
    height: 360,
    margin: { t: 20, r: 70, b: 40, l: 130 },
    xaxis: Object.assign({}, LAYOUT_BASE.xaxis, {
      title: { text: 'SHAP value', font: { size: 11, color: COLORS.muted } },
      zeroline: true,
      zerolinecolor: '#d1d5db',
      zerolinewidth: 2,
    }),
    yaxis: Object.assign({}, LAYOUT_BASE.yaxis, {
      tickfont: { size: 10 },
    }),
  });

  Plotly.newPlot(el, [trace], layout, CONFIG);
}

/* ═══════════════════════════════════════════════════════════════════════
   Global Feature Importance (RF feature_importances_)
═══════════════════════════════════════════════════════════════════════ */
function renderGlobalImportance(data) {
  const el = document.getElementById('globalImportance');
  if (!el || typeof Plotly === 'undefined') return;

  const paired = data.feature_names.map((n, i) => ({
    name: fmtFeature(n),
    val:  data.feature_importances[i],
  }));
  paired.sort((a, b) => a.val - b.val);

  // Gradient of blue shades by importance
  const maxVal = Math.max(...paired.map(p => p.val));
  const getColor = (v) => {
    const alpha = 0.25 + (v / maxVal) * 0.75;
    return `rgba(37,99,235,${alpha.toFixed(2)})`;
  };

  const trace = {
    type: 'bar',
    orientation: 'h',
    x: paired.map(p => p.val),
    y: paired.map(p => p.name),
    marker: {
      color: paired.map(p => getColor(p.val)),
      line: { color: COLORS.blue, width: 0.5 },
    },
    text: paired.map(p => (p.val * 100).toFixed(1) + '%'),
    textposition: 'outside',
    textfont: { size: 10, color: COLORS.blue, family: 'DM Mono, monospace' },
  };

  const layout = Object.assign({}, LAYOUT_BASE, {
    height: 360,
    margin: { t: 20, r: 60, b: 40, l: 130 },
    xaxis: Object.assign({}, LAYOUT_BASE.xaxis, {
      title: { text: 'Importance', font: { size: 11, color: COLORS.muted } },
    }),
    yaxis: Object.assign({}, LAYOUT_BASE.yaxis, {
      tickfont: { size: 10 },
    }),
  });

  Plotly.newPlot(el, [trace], layout, CONFIG);
}

/* ═══════════════════════════════════════════════════════════════════════
   Model Comparison Chart (Research page)
═══════════════════════════════════════════════════════════════════════ */
function renderModelComparison(elId, paperRF, paperLR, deployedRF, deployedLR) {
  const el = document.getElementById(elId);
  if (!el || typeof Plotly === 'undefined') return;

  const metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score'];
  const keys    = ['accuracy', 'precision', 'recall', 'f1'];

  const traces = [
    {
      name: 'Paper — Random Forest (30 features)',
      type: 'bar',
      x: metrics,
      y: keys.map(k => paperRF[k]),
      marker: { color: 'rgba(37,99,235,0.85)', line: { color: '#1D4ED8', width: 1 } },
      text: keys.map(k => paperRF[k] + '%'),
      textposition: 'outside',
      textfont: { size: 10, family: 'DM Mono, monospace' },
    },
    {
      name: 'Paper — Logistic Regression (30 features)',
      type: 'bar',
      x: metrics,
      y: keys.map(k => paperLR[k]),
      marker: { color: 'rgba(37,99,235,0.35)', line: { color: '#93C5FD', width: 1 } },
      text: keys.map(k => paperLR[k] + '%'),
      textposition: 'outside',
      textfont: { size: 10, family: 'DM Mono, monospace' },
    },
    {
      name: 'Deployed — Random Forest (16 features)',
      type: 'bar',
      x: metrics,
      y: keys.map(k => deployedRF[k]),
      marker: { color: 'rgba(13,148,136,0.85)', line: { color: '#0F766E', width: 1 } },
      text: keys.map(k => deployedRF[k] + '%'),
      textposition: 'outside',
      textfont: { size: 10, family: 'DM Mono, monospace' },
    },
    {
      name: 'Deployed — Logistic Regression (16 features)',
      type: 'bar',
      x: metrics,
      y: keys.map(k => deployedLR[k]),
      marker: { color: 'rgba(13,148,136,0.35)', line: { color: '#99F6E4', width: 1 } },
      text: keys.map(k => deployedLR[k] + '%'),
      textposition: 'outside',
      textfont: { size: 10, family: 'DM Mono, monospace' },
    },
  ];

  const layout = Object.assign({}, LAYOUT_BASE, {
    barmode: 'group',
    height: 380,
    margin: { t: 20, r: 20, b: 60, l: 60 },
    xaxis: Object.assign({}, LAYOUT_BASE.xaxis, {
      tickfont: { size: 12 },
    }),
    yaxis: Object.assign({}, LAYOUT_BASE.yaxis, {
      title: { text: 'Score (%)', font: { size: 11, color: COLORS.muted } },
      range: [85, 100],
    }),
    legend: {
      orientation: 'h',
      x: 0, y: -0.25,
      font: { size: 10 },
    },
  });

  Plotly.newPlot(el, traces, layout, CONFIG);
}
