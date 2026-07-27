document.addEventListener('DOMContentLoaded', () => {
    fetchPerformance();
    fetchPipelineStatus();

    // Poll performance every 10s, pipeline status every 15s
    setInterval(fetchPerformance, 10000);
    setInterval(fetchPipelineStatus, 15000);

    // ── Retrain button ───────────────────────────────────────────────
    const btnRetrain = document.getElementById('btn-retrain');
    btnRetrain.addEventListener('click', () => {
        const statusDiv = document.getElementById('retrain-status');
        const icon = btnRetrain.querySelector('i');

        btnRetrain.disabled = true;
        icon.setAttribute('data-feather', 'loader');
        icon.classList.add('spinner');
        feather.replace();

        statusDiv.classList.remove('hidden');
        statusDiv.style.display = 'flex';
        statusDiv.style.color = 'var(--text-secondary)';
        statusDiv.textContent = 'Triggering workflow...';

        fetch('/admin/retrain', { method: 'POST' })
            .then(res => res.json().then(data => ({ status: res.status, body: data })))
            .then(({ status, body }) => {
                btnRetrain.disabled = false;
                icon.setAttribute('data-feather', 'play');
                icon.classList.remove('spinner');
                feather.replace();

                if (status === 200) {
                    statusDiv.style.color = 'var(--success-color)';
                    statusDiv.innerHTML = '<i data-feather="check"></i> Pipeline triggered! Refreshing status...';
                    // Poll more aggressively right after trigger
                    setTimeout(fetchPipelineStatus, 3000);
                    setTimeout(fetchPipelineStatus, 8000);
                    setTimeout(fetchPipelineStatus, 20000);
                } else {
                    statusDiv.style.color = 'var(--warning-color)';
                    statusDiv.textContent = body.message || 'Workflow triggered (check logs).';
                }
                feather.replace();
            })
            .catch(err => {
                btnRetrain.disabled = false;
                icon.setAttribute('data-feather', 'play');
                icon.classList.remove('spinner');
                feather.replace();
                statusDiv.style.color = 'var(--danger-color)';
                statusDiv.innerHTML = '<i data-feather="alert-triangle"></i> Error: ' + err.message;
                feather.replace();
            });
    });

    // ── Data upload ──────────────────────────────────────────────────
    const dataDrop = document.getElementById('data-drop-zone');
    const dataInput = document.getElementById('data-file-input');
    const uploadStatus = document.getElementById('upload-status');

    dataDrop.addEventListener('dragover', (e) => {
        e.preventDefault();
        dataDrop.classList.add('dragover');
    });
    ['dragleave', 'dragend'].forEach(type => {
        dataDrop.addEventListener(type, () => dataDrop.classList.remove('dragover'));
    });
    dataDrop.addEventListener('drop', (e) => {
        e.preventDefault();
        dataDrop.classList.remove('dragover');
        if (e.dataTransfer.files.length) uploadData(e.dataTransfer.files[0]);
    });
    dataInput.addEventListener('change', (e) => {
        if (e.target.files.length) uploadData(e.target.files[0]);
    });

    function uploadData(file) {
        if (!file.name.endsWith('.zip')) {
            uploadStatus.classList.remove('hidden');
            uploadStatus.style.color = 'var(--danger-color)';
            uploadStatus.innerHTML = '<i data-feather="x-circle"></i> Only .zip files are supported.';
            feather.replace();
            return;
        }
        uploadStatus.classList.remove('hidden');
        uploadStatus.style.color = 'var(--text-secondary)';
        uploadStatus.innerHTML = '<i data-feather="loader" class="spinner"></i> Uploading...';
        feather.replace();

        const formData = new FormData();
        formData.append('file', file);

        fetch('/admin/data', { method: 'POST', body: formData })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    uploadStatus.style.color = 'var(--success-color)';
                    uploadStatus.innerHTML = `<i data-feather="check-circle"></i> Uploaded: ${data.path}`;
                    feather.replace();
                } else {
                    throw new Error(data.detail || 'Upload failed');
                }
            })
            .catch(err => {
                uploadStatus.style.color = 'var(--danger-color)';
                uploadStatus.innerHTML = `<i data-feather="alert-circle"></i> ${err.message}`;
                feather.replace();
            });
    }

    // ── Pipeline Status ──────────────────────────────────────────────
    function fetchPipelineStatus() {
        fetch('/admin/pipeline-status')
            .then(res => res.json())
            .then(data => {
                const dot = document.getElementById('pipeline-dot');
                const text = document.getElementById('pipeline-status-text');
                const meta = document.getElementById('pipeline-meta');
                const link = document.getElementById('pipeline-link');

                const conclusionColors = {
                    success: '#10b981',
                    failure: '#ef4444',
                    cancelled: '#6b7280',
                    skipped: '#6b7280',
                };
                const conclusionIcons = {
                    success: '✅', failure: '❌', cancelled: '⛔', skipped: '⏭️'
                };

                let dotColor = '#6b7280';
                let label = 'Unknown';

                if (data.status === 'in_progress') {
                    dotColor = '#f59e0b';
                    label = '⏳ Running…';
                    dot.style.boxShadow = '0 0 0 3px rgba(245,158,11,0.3)';
                } else if (data.status === 'queued') {
                    dotColor = '#60a5fa';
                    label = '🕐 Queued';
                    dot.style.boxShadow = '';
                } else if (data.status === 'completed') {
                    const c = data.conclusion || 'unknown';
                    dotColor = conclusionColors[c] || '#6b7280';
                    label = `${conclusionIcons[c] || '•'} ${c.charAt(0).toUpperCase() + c.slice(1)}`;
                    dot.style.boxShadow = '';
                } else if (data.status === 'none') {
                    label = 'No runs yet';
                    dot.style.boxShadow = '';
                }

                dot.style.background = dotColor;
                text.textContent = label;

                if (data.started_at) {
                    const d = new Date(data.started_at);
                    meta.textContent = `Run #${data.run_number} · ${d.toLocaleString()} · ${data.branch || ''}`;
                } else if (data.message) {
                    meta.textContent = data.message;
                }

                if (data.url) {
                    link.href = data.url;
                    link.style.display = 'inline';
                } else {
                    link.style.display = 'none';
                }
            })
            .catch(() => {
                const text = document.getElementById('pipeline-status-text');
                if (text) text.textContent = 'Unable to fetch status';
            });
    }

    // ── Performance / Feedback ───────────────────────────────────────
    function fetchPerformance() {
        fetch('/performance')
            .then(res => res.json())
            .then(data => {
                const acc = document.getElementById('stat-accuracy');
                const count = document.getElementById('stat-feedback-count');
                const confPairs = document.getElementById('confusion-pairs');

                count.textContent = data.total_feedback;

                if (data.total_feedback === 0) {
                    acc.textContent = 'N/A';
                    confPairs.innerHTML = '<div style="color: var(--text-secondary); text-align: center; padding: 2rem 0;">No feedback data yet.</div>';
                } else {
                    const pct = (data.accuracy * 100).toFixed(1);
                    acc.textContent = pct + '%';
                    acc.style.color = data.accuracy > 0.6 ? 'var(--success-color)' : 'var(--warning-color)';

                    if (data.top_confusion_pairs && data.top_confusion_pairs.length > 0) {
                        let html = '';
                        data.top_confusion_pairs.forEach(pair => {
                            html += `
                                <div class="feedback-item">
                                    <div class="feedback-flow">
                                        <div class="class-tag class-wrong">${pair.predicted}</div>
                                        <i data-feather="arrow-right" style="color: var(--text-secondary); width: 16px;"></i>
                                        <div class="class-tag class-right">${pair.actual}</div>
                                    </div>
                                    <div class="feedback-count">${pair.count}</div>
                                </div>
                            `;
                        });
                        confPairs.innerHTML = html;
                        feather.replace();
                    } else {
                        confPairs.innerHTML = '<div style="color: var(--text-secondary); text-align: center; padding: 2rem 0;">No misclassifications recorded. Perfect!</div>';
                    }
                }
            })
            .catch(console.error);
    }
});
