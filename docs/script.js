// Interactive MNIST Classifier JS Logic

// Model Weights Container
let modelWeights = null;

// DOM Elements
const canvas = document.getElementById('drawing-canvas');
const ctx = canvas.getContext('2d');
const previewCanvas = document.getElementById('preview-canvas');
const pCtx = previewCanvas.getContext('2d');
const btnClear = document.getElementById('btn-clear');
const btnToggleGrid = document.getElementById('btn-toggle-grid');
const gridOverlay = document.getElementById('grid-overlay');
const predictedDigitEl = document.getElementById('predicted-digit');
const predictionConfidenceEl = document.getElementById('prediction-confidence');
const chartContainer = document.getElementById('probabilities-chart');

// Tab navigation elements
const tabButtons = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

// Status table elements
const w1Status = document.getElementById('w1-status');
const b1Status = document.getElementById('b1-status');
const w2Status = document.getElementById('w2-status');
const b2Status = document.getElementById('b2-status');
const w3Status = document.getElementById('w3-status');
const b3Status = document.getElementById('b3-status');

// Drawing state
let isDrawing = false;
let lastX = 0;
let lastY = 0;
let hasDrawn = false;

// ────────────────────────────────────────────────────────────────
// 1. Initialization and UI Events
// ────────────────────────────────────────────────────────────────

// Set up drawing settings
ctx.lineJoin = 'round';
ctx.lineCap = 'round';
ctx.lineWidth = 18; // Brush thickness
ctx.strokeStyle = '#ffffff';

// Initialize probability chart
const chartRows = [];
function initChart() {
    chartContainer.innerHTML = '';
    for (let i = 0; i < 10; i++) {
        const row = document.createElement('div');
        row.className = 'chart-row';
        row.innerHTML = `
            <span class="digit-label">${i}</span>
            <div class="bar-wrapper">
                <div class="bar-fill" id="bar-fill-${i}"></div>
            </div>
            <span class="bar-value" id="bar-val-${i}">0.0%</span>
        `;
        chartContainer.appendChild(row);
        chartRows.push({
            row: row,
            fill: document.getElementById(`bar-fill-${i}`),
            val: document.getElementById(`bar-val-${i}`)
        });
    }
}

// Reset predictions and chart displays
function resetOutputs() {
    predictedDigitEl.textContent = '—';
    predictionConfidenceEl.textContent = '0.0%';
    predictionConfidenceEl.style.color = 'var(--text-secondary)';
    
    chartRows.forEach(rowInfo => {
        rowInfo.row.classList.remove('active');
        rowInfo.fill.style.width = '0%';
        rowInfo.val.textContent = '0.0%';
    });
}

// Clear the canvases
function clearCanvas() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    pCtx.fillStyle = '#000000';
    pCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
    
    hasDrawn = false;
    resetOutputs();
}

// Set up drawing event listeners for Mouse & Touch
function getCoordinates(e) {
    const rect = canvas.getBoundingClientRect();
    if (e.touches && e.touches.length > 0) {
        return {
            x: e.touches[0].clientX - rect.left,
            y: e.touches[0].clientY - rect.top
        };
    } else {
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }
}

function startDrawing(e) {
    isDrawing = true;
    const coords = getCoordinates(e);
    lastX = coords.x;
    lastY = coords.y;
    hasDrawn = true;
    draw(e);
}

function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    
    const coords = getCoordinates(e);
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(coords.x, coords.y);
    ctx.stroke();
    
    lastX = coords.x;
    lastY = coords.y;
    
    runInference();
}

function stopDrawing() {
    isDrawing = false;
}

// Add event listeners to drawing canvas
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

canvas.addEventListener('touchstart', startDrawing, { passive: false });
canvas.addEventListener('touchmove', draw, { passive: false });
canvas.addEventListener('touchend', stopDrawing);

// Control Buttons
btnClear.addEventListener('click', clearCanvas);

btnToggleGrid.addEventListener('click', () => {
    gridOverlay.classList.toggle('hidden');
    btnToggleGrid.classList.toggle('btn-secondary');
    if (!gridOverlay.classList.contains('hidden')) {
        btnToggleGrid.style.background = 'var(--accent-glow)';
        btnToggleGrid.style.borderColor = 'rgba(59, 130, 246, 0.4)';
    } else {
        btnToggleGrid.style.background = '';
        btnToggleGrid.style.borderColor = '';
    }
});

// Setup tab navigation
tabButtons.forEach(button => {
    button.addEventListener('click', () => {
        const tabId = button.getAttribute('data-tab');
        
        tabButtons.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));
        
        button.classList.add('active');
        document.getElementById(tabId).classList.add('active');
    });
});

// Try to resolve correct github repository link dynamically from browser url
try {
    const pathParts = window.location.pathname.split('/');
    if (window.location.hostname.includes('github.io')) {
        const user = window.location.hostname.split('.')[0];
        const repo = pathParts[1] || 'mnist-ann-classifier';
        document.getElementById('github-link').href = `https://github.com/${user}/${repo}`;
    }
} catch (e) {
    // Fallback to generic URL
}

// ────────────────────────────────────────────────────────────────
// 2. Weights Fetching
// ────────────────────────────────────────────────────────────────

function setLayerStatus(el, success, shapeStr) {
    if (success) {
        el.innerHTML = `<span class="status-indicator success">Loaded (${shapeStr})</span>`;
    } else {
        el.innerHTML = `<span class="status-indicator warning">Failed</span>`;
    }
}

async function loadWeights() {
    try {
        const response = await fetch('model_weights.json');
        if (!response.ok) {
            throw new Error(`Failed to load model weights: ${response.statusText}`);
        }
        modelWeights = await response.json();
        
        // Update diagnostics table
        setLayerStatus(w1Status, true, '784 × 128');
        setLayerStatus(b1Status, true, '128');
        setLayerStatus(w2Status, true, '128 × 64');
        setLayerStatus(b2Status, true, '64');
        setLayerStatus(w3Status, true, '64 × 10');
        setLayerStatus(b3Status, true, '10');
        
        console.log('Model weights successfully loaded.');
    } catch (error) {
        console.error('Error loading model weights:', error);
        [w1Status, b1Status, w2Status, b2Status, w3Status, b3Status].forEach(el => {
            setLayerStatus(el, false);
        });
        
        // Render error message inside predictions panel
        predictedDigitEl.textContent = '❌';
        predictionConfidenceEl.textContent = 'Error loading weights';
        predictionConfidenceEl.style.color = 'var(--danger)';
    }
}

// ────────────────────────────────────────────────────────────────
// 3. Preprocessing and Neural Network Inference
// ────────────────────────────────────────────────────────────────

// Activation Functions
function relu(sum) {
    return Math.max(0, sum);
}

function softmax(z) {
    const maxVal = Math.max(...z); // For numerical stability
    const exps = z.map(val => Math.exp(val - maxVal));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(val => val / sumExps);
}

// Matrix multiplication and bias summation
function denseLayer(x, W, b, activationFn) {
    const outDim = b.length;
    const inDim = x.length;
    const result = new Float32Array(outDim);
    
    for (let i = 0; i < outDim; i++) {
        let sum = b[i];
        for (let j = 0; j < inDim; j++) {
            sum += x[j] * W[j][i];
        }
        
        if (activationFn === 'relu') {
            result[i] = relu(sum);
        } else {
            result[i] = sum; // Return raw values for subsequent Softmax
        }
    }
    return result;
}

// Subsamples the 280x280 canvas to 28x28 grayscale vector
function preprocessImage() {
    // 1. Draw downscaled version of main canvas to the 28x28 preview canvas
    pCtx.fillStyle = '#000000';
    pCtx.fillRect(0, 0, 28, 28);
    pCtx.drawImage(canvas, 0, 0, 28, 28);
    
    // 2. Fetch the 28x28 image pixels
    const imgData = pCtx.getImageData(0, 0, 28, 28);
    const pixels = imgData.data;
    
    // 3. Convert RGBA to Grayscale normalized in [0, 1]
    const inputVector = new Float32Array(784);
    for (let i = 0; i < 784; i++) {
        // Red channel is at index i*4
        // Since we draw white on black, R=G=B. Grayscale intensity is just the R value.
        inputVector[i] = pixels[i * 4] / 255.0;
    }
    
    return inputVector;
}

function runInference() {
    if (!modelWeights || !hasDrawn) return;
    
    // 1. Preprocess canvas drawing to input vector
    const x = preprocessImage();
    
    // 2. Forward Propagation
    // Layer 1 (Dense 128, ReLU)
    const h1 = denseLayer(x, modelWeights.W1, modelWeights.b1, 'relu');
    
    // Layer 2 (Dense 64, ReLU)
    const h2 = denseLayer(h1, modelWeights.W2, modelWeights.b2, 'relu');
    
    // Layer 3 (Dense 10, Logits)
    const logits = denseLayer(h2, modelWeights.W3, modelWeights.b3, 'linear');
    
    // Apply Softmax to logits to get probabilities
    const probabilities = softmax(Array.from(logits));
    
    // 3. Find predicted digit (highest probability)
    let predictedDigit = 0;
    let maxProb = -1;
    for (let i = 0; i < 10; i++) {
        if (probabilities[i] > maxProb) {
            maxProb = probabilities[i];
            predictedDigit = i;
        }
    }
    
    // 4. Update UI displays
    predictedDigitEl.textContent = predictedDigit;
    
    const confidencePercent = (maxProb * 100).toFixed(1);
    predictionConfidenceEl.textContent = `${confidencePercent}%`;
    
    // Dynamic color coding for confidence
    if (maxProb > 0.8) {
        predictionConfidenceEl.style.color = 'var(--success)';
    } else if (maxProb > 0.5) {
        predictionConfidenceEl.style.color = '#fbbf24'; // Warning yellow
    } else {
        predictionConfidenceEl.style.color = 'var(--danger)';
    }
    
    // Update chart rows
    probabilities.forEach((prob, i) => {
        const percent = (prob * 100).toFixed(1);
        chartRows[i].fill.style.width = `${percent}%`;
        chartRows[i].val.textContent = `${percent}%`;
        
        if (i === predictedDigit) {
            chartRows[i].row.classList.add('active');
        } else {
            chartRows[i].row.classList.remove('active');
        }
    });
}

// Initialize on page load
initChart();
clearCanvas();
loadWeights();
