# Dokumentations-Metriken

Diese Seite visualisiert die Qualität und Abdeckung unserer Dokumentation.

<div id="metrics-dashboard">
  <table>
    <thead>
      <tr>
        <th>Metrik</th>
        <th>Status / Wert</th>
        <th>Quelle</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>API Doc Abdeckung</td>
        <td><img src="../assets/interrogate.svg" alt="Interrogate Coverage"></td>
        <td><code>interrogate</code></td>
      </tr>
      <tr>
        <td>Defekte Links</td>
        <td id="metric-links">Wird geladen...</td>
        <td><code>lychee</code></td>
      </tr>
      <tr>
        <td>Markdown Lint</td>
        <td id="metric-lint">Wird geladen...</td>
        <td><code>markdownlint</code></td>
      </tr>
    </tbody>
  </table>
</div>

<script>
fetch('../assets/metrics.json')
  .then(response => response.json())
  .then(data => {
    document.getElementById('metric-links').textContent = data.broken_links === 0 ? '✅ Keine' : '❌ ' + data.broken_links;
    document.getElementById('metric-lint').textContent = data.lint_errors === 0 ? '✅ 0 Fehler' : '⚠️ ' + data.lint_errors;
  })
  .catch(err => {
    console.log('Metrics not yet available');
  });
</script>
