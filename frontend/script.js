const teamEl = document.getElementById('team');
const seasonEl = document.getElementById('season');
const lastEl = document.getElementById('last');
const oppEl = document.getElementById('opponent');
const ingestBtn = document.getElementById('ingest');
const recommendBtn = document.getElementById('recommend');
const resultEl = document.getElementById('result');

const api = (path, options = {}) =>
  fetch(`http://localhost:8000${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });

function showMessage(msg) {
  resultEl.innerHTML = `<pre>${msg}</pre>`;
}

ingestBtn.onclick = async () => {
  showMessage('Ingesting...');
  try {
    const body = {
      team: teamEl.value.trim(),
      season: seasonEl.value.trim(),
      last: Number(lastEl.value) || 10,
    };
    const res = await api('/ingest', { method: 'POST', body: JSON.stringify(body) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'ingest failed');
    showMessage(`Done: added ${data.docs_added} documents`);
  } catch (e) {
    showMessage(`Error: ${e.message}`);
  }
};

recommendBtn.onclick = async () => {
  showMessage('Generating...');
  try {
    const body = {
      team: teamEl.value.trim(),
      season: seasonEl.value.trim(),
      opponent: oppEl.value.trim(),
    };
    const res = await api('/recommend', { method: 'POST', body: JSON.stringify(body) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'recommend failed');
    renderResult(data);
  } catch (e) {
    showMessage(`Error: ${e.message}`);
  }
};

function renderResult(data) {
  const { suggested_lineup = [], supporting_games = [] } = data;
  const lineupCards = suggested_lineup
    .map(
      (p) => `
      <div class="card">
        <h3>${p.player}</h3>
        <div class="small">Minutes ${p.avg_minutes} | PTS ${p.avg_pts} | REB ${p.avg_reb} | AST ${p.avg_ast}</div>
        ${p.opponent_history ? '<span class="badge">Has opponent sample</span>' : ''}
      </div>
    `,
    )
    .join('');

  const snippetCards = supporting_games
    .map(
      (g) => `
      <div class="card">
        <h4>${g.player}</h4>
        <div class="small">${g.game_date} vs ${g.opponent}</div>
        <div class="small">${g.minutes} MIN, ${g.pts} PTS</div>
      </div>
    `,
    )
    .join('');

  resultEl.innerHTML = `
    <h2>Suggested Rotation</h2>
    <div class="result-grid">${lineupCards || '<div class="small">No data</div>'}</div>
    <h2>Supporting Samples</h2>
    <div class="result-grid">${snippetCards || '<div class="small">No data</div>'}</div>
  `;
}
