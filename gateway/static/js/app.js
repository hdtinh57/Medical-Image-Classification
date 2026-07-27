const CLASS_NAMES = [
    "actinic keratosis",
    "basal cell carcinoma",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "pigmented benign keratosis",
    "seborrheic keratosis",
    "squamous cell carcinoma",
    "vascular lesion"
];

let currentPredictionId = null;
let currentPredictedClass = null;

document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreviewSrc = document.getElementById('image-preview-src');
    const spinner = document.getElementById('loading');
    const resultPanel = document.getElementById('result-panel');
    const emptyState = document.getElementById('empty-state');
    const probsChart = document.getElementById('probs-chart');

    // Populate actual class select
    const select = document.getElementById('actual-class-select');
    CLASS_NAMES.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c;
        opt.textContent = c;
        select.appendChild(opt);
    });

    // Handle drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    ['dragleave', 'dragend'].forEach(type => {
        dropZone.addEventListener(type, () => dropZone.classList.remove('dragover'));
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        // Reset UI
        previewContainer.classList.remove('hidden');
        resultPanel.classList.add('hidden');
        emptyState.classList.add('hidden');
        resetFeedbackUI();
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreviewSrc.src = e.target.result;
        };
        reader.readAsDataURL(file);

        // Upload
        const formData = new FormData();
        formData.append('file', file);
        
        spinner.classList.remove('hidden');
        dropZone.classList.add('hidden');

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) throw new Error('Prediction failed');
            return response.json();
        })
        .then(data => {
            spinner.classList.add('hidden');
            dropZone.classList.remove('hidden');
            showResults(data);
        })
        .catch(error => {
            spinner.classList.add('hidden');
            dropZone.classList.remove('hidden');
            alert('Error calling predict API: ' + error.message);
            emptyState.classList.remove('hidden');
        });
    }

    function showResults(data) {
        currentPredictionId = data.prediction_id;
        currentPredictedClass = data.predicted_class;

        document.getElementById('predicted-class').textContent = data.predicted_class;
        document.getElementById('confidence-text').textContent = (data.confidence * 100).toFixed(2) + '%';
        
        setTimeout(() => {
            document.getElementById('confidence-fill').style.width = (data.confidence * 100) + '%';
        }, 100);
        
        document.getElementById('latency-text').textContent = data.latency_seconds;

        // Render probabilities
        probsChart.innerHTML = '';
        const sortedProbs = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
        
        sortedProbs.slice(0, 5).forEach(([cls, prob]) => {
            const isTop = cls === data.predicted_class;
            const pct = (prob * 100).toFixed(1);
            
            const html = `
                <div class="prob-row">
                    <div class="prob-label" title="${cls}">${cls}</div>
                    <div class="prob-bar-container">
                        <div class="prob-bar-fill ${isTop ? 'top' : ''}" style="width: ${pct}%"></div>
                    </div>
                    <div class="prob-value">${pct}%</div>
                </div>
            `;
            probsChart.insertAdjacentHTML('beforeend', html);
        });

        resultPanel.classList.remove('hidden');
    }

    // Feedback logic
    document.getElementById('btn-correct').addEventListener('click', () => {
        submitFeedback(currentPredictedClass);
    });

    document.getElementById('btn-incorrect').addEventListener('click', () => {
        document.getElementById('feedback-buttons').classList.add('hidden');
        document.getElementById('correction-form').classList.remove('hidden');
        // Preset select to something else
        const select = document.getElementById('actual-class-select');
        for (let i = 0; i < select.options.length; i++) {
            if (select.options[i].value !== currentPredictedClass) {
                select.selectedIndex = i;
                break;
            }
        }
    });

    document.getElementById('btn-submit-correction').addEventListener('click', () => {
        const actual = document.getElementById('actual-class-select').value;
        submitFeedback(actual);
    });

    function submitFeedback(actualClass) {
        if (!currentPredictionId) return;

        fetch('/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                prediction_id: currentPredictionId,
                actual_class: actualClass
            })
        })
        .then(res => {
            if (res.ok) {
                document.getElementById('feedback-buttons').classList.add('hidden');
                document.getElementById('correction-form').classList.add('hidden');
                document.getElementById('feedback-success').classList.remove('hidden');
            } else {
                alert('Failed to submit feedback.');
            }
        });
    }

    function resetFeedbackUI() {
        document.getElementById('feedback-buttons').classList.remove('hidden');
        document.getElementById('correction-form').classList.add('hidden');
        document.getElementById('feedback-success').classList.add('hidden');
        document.getElementById('confidence-fill').style.width = '0%';
    }
});
