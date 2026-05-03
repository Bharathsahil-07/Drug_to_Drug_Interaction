// State management
let selectedDrugs = [];
let allDrugs = [];
let searchTimeout;
let cy; // Cytoscape instance

// API base URL
const API_URL = window.location.origin;

// Initialize
document.addEventListener('DOMContentLoaded', function () {
    setupEventListeners();
    fetchInsights(); // Initial fetch
    setInterval(fetchInsights, 5000); // Polling every 5s
});

// Fetch Insights
async function fetchInsights() {
    try {
        const response = await fetch(`${API_URL}/api/insights`);
        const data = await response.json();

        const elTotalDrugs = document.getElementById('total-drugs');
        if (elTotalDrugs) elTotalDrugs.textContent = data.total_drugs;

        const elTotalInteractions = document.getElementById('total-interactions');
        if (elTotalInteractions) elTotalInteractions.textContent = data.total_interactions.toLocaleString();

        const elModelAcc = document.getElementById('model-acc');
        if (elModelAcc) elModelAcc.textContent = (data.accuracy * 100).toFixed(1) + '%';

        const elLastUpdate = document.getElementById('last-update');
        if (elLastUpdate) elLastUpdate.textContent = new Date().toLocaleTimeString();

        const logList = document.getElementById('activity-log');
        if (logList && data.recent_activity) {
            logList.innerHTML = '';
            data.recent_activity.forEach(log => {
                const li = document.createElement('li');
                li.className = 'list-group-item d-flex justify-content-between align-items-center py-1';
                // Handle both old and new log formats if necessary
                const drug1 = log.drug1 || log.drug1_name || '?';
                const drug2 = log.drug2 || log.drug2_name || '?';
                const pred = log.prediction || (log.risk_level ? `${log.risk_level} Risk` : 'Checked');
                const isRisk = pred.includes('Possible') || pred.includes('HIGH') || pred.includes('Major');

                li.innerHTML = `
                    <span>${drug1} + ${drug2}</span>
                    <span class="badge ${isRisk ? 'bg-danger' : 'bg-success'} rounded-pill" style="font-size: 0.7rem;">${pred}</span>
                `;
                logList.appendChild(li);
            });
        }
    } catch (e) {
        console.error("Failed to fetch insights", e);
    }
}

// Setup event listeners
function setupEventListeners() {
    const searchInput = document.getElementById('drugSearch');
    const checkBtn = document.getElementById('checkBtn');

    if (searchInput) {
        // Drug search with debounce
        searchInput.addEventListener('input', function () {
            clearTimeout(searchTimeout);
            const query = this.value.trim();

            if (query.length < 2) {
                const results = document.getElementById('drugResults');
                if (results) results.style.display = 'none';
                return;
            }

            searchTimeout = setTimeout(() => searchDrugs(query), 300);
        });
    }

    // Check interactions button
    if (checkBtn) {
        checkBtn.addEventListener('click', checkInteractions);
    }
}

// Search drugs
async function searchDrugs(query) {
    try {
        const response = await fetch(`${API_URL}/api/drugs/search?q=${encodeURIComponent(query)}&limit=20`);
        const data = await response.json();

        const resultsDiv = document.getElementById('drugResults');
        if (!resultsDiv) return;

        if (data.drugs.length === 0) {
            resultsDiv.innerHTML = '<div class="drug-item">No drugs found</div>';
        } else {
            resultsDiv.innerHTML = data.drugs.map(drug => `
                <div class="drug-item" onclick="selectDrug('${drug.drug_id}', '${drug.name.replace(/'/g, "\\'")}')">
                    <i class="fas fa-pills"></i> ${drug.name}
                    <small class="text-muted">(${drug.drug_id})</small>
                </div>
            `).join('');
        }

        resultsDiv.style.display = 'block';
    } catch (error) {
        console.error('Error searching drugs:', error);
    }
}

// Select drug
window.selectDrug = function (drugId, drugName) {
    // Check if already selected
    if (selectedDrugs.some(d => d.id === drugId)) {
        alert('This drug is already selected!');
        return;
    }

    selectedDrugs.push({ id: drugId, name: drugName });
    updateSelectedDrugs();

    // Clear search
    const searchInput = document.getElementById('drugSearch');
    const resultsDiv = document.getElementById('drugResults');
    if (searchInput) searchInput.value = '';
    if (resultsDiv) resultsDiv.style.display = 'none';
}

// Remove drug
window.removeDrug = function (drugId) {
    selectedDrugs = selectedDrugs.filter(d => d.id !== drugId);
    updateSelectedDrugs();
}

// Update selected drugs display
function updateSelectedDrugs() {
    const container = document.getElementById('selectedDrugs');
    const checkBtn = document.getElementById('checkBtn');
    if (!container) return;

    if (selectedDrugs.length === 0) {
        container.innerHTML = `
            <p class="text-muted text-center" style="margin: 40px 0;">
                <i class="fas fa-hand-pointer fa-2x"></i><br>
                Select at least 2 drugs
            </p>
        `;
        if (checkBtn) checkBtn.disabled = true;
    } else {
        container.innerHTML = selectedDrugs.map(drug => `
            <div class="drug-tag">
                <i class="fas fa-pills"></i> ${drug.name}
                <button class="remove-btn" onclick="removeDrug('${drug.id}')">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `).join('');

        if (checkBtn) checkBtn.disabled = selectedDrugs.length < 2;
    }
}

// Initialize/Update Graph
function updateGraph(drugs, interactions) {
    const cyContainer = document.getElementById('cy');
    if (!cyContainer) return;

    const elements = [];
    const nodeSet = new Set();
    const tooltip = document.getElementById('graphTooltip');
    const explanationCard = document.getElementById('graphExplanation');
    const explanationText = document.getElementById('graphExplanationText');

    // Helper to add nodes safely
    const addNode = (id, name, isTarget) => {
        if (!nodeSet.has(id)) {
            elements.push({
                data: {
                    id: id,
                    label: name,
                    type: isTarget ? 'target' : 'neighbor'
                }
            });
            nodeSet.add(id);
        }
    };

    // Add selected target drugs
    drugs.forEach(d => addNode(d.id, d.name, true));

    // Add edges and neighbors from interactions
    if (interactions) {
        interactions.forEach(res => {
            const d1 = res.drug1;
            const d2 = res.drug2;

            if (!d1 || !d2) return;

            // Ensure target nodes exist
            addNode(d1.id, d1.name, true);
            addNode(d2.id, d2.name, true);

            const prob = res.probability || 0;
            const isHigh = res.risk_level === 'HIGH' || res.severity === 'Major' || prob > 0.7;
            const isMod = res.risk_level === 'MEDIUM' || res.severity === 'Moderate' || prob > 0.4;

            let color = '#4caf50'; // Safe/Low
            if (isHigh) color = '#d32f2f';
            else if (isMod) color = '#f57c00';

            // Main interaction edge
            elements.push({
                data: {
                    id: `edge-${d1.id}-${d2.id}`,
                    source: d1.id,
                    target: d2.id,
                    probability: prob,
                    risk: res.risk_level,
                    description: res.description,
                    mechanism: res.clinical_explanation?.mechanism || res.description
                },
                style: {
                    'line-color': color,
                    'target-arrow-color': color,
                    'width': Math.max(2, prob * 8),
                    'line-style': isHigh ? 'solid' : 'dashed'
                }
            });

            // Add Top Influencing Neighbors as context
            if (res.model_explanation && res.model_explanation.top_influencing_neighbors) {
                res.model_explanation.top_influencing_neighbors.slice(0, 3).forEach(n => {
                    const neighborId = n.drug_id || `n-${n.drug}`;
                    addNode(neighborId, n.drug, false);

                    // Edge to neighbor (representing influence/similarity in GAT)
                    elements.push({
                        data: {
                            id: `inf-${neighborId}-${d1.id}`,
                            source: neighborId,
                            target: d1.id,
                            influence: n.relative_influence,
                            type: 'influence'
                        },
                        style: {
                            'line-color': '#a5a5a5',
                            'width': 1,
                            'line-style': 'dotted',
                            'target-arrow-shape': 'none'
                        }
                    });
                });
            }
        });
    }

    if (cy) {
        cy.json({ elements: elements });
        cy.layout({ name: 'cose', padding: 30, animate: true, nodeRepulsion: 4000 }).run();
    } else {
        // Initialize Cytoscape
        cy = cytoscape({
            container: cyContainer,
            elements: elements,
            style: [
                {
                    selector: 'node',
                    style: {
                        'background-color': '#667eea',
                        'label': 'data(label)',
                        'color': '#333',
                        'font-size': '10px',
                        'text-valign': 'bottom',
                        'text-margin-y': '5px',
                        'width': '40px',
                        'height': '40px',
                        'text-outline-width': 2,
                        'text-outline-color': '#fff',
                        'transition-property': 'background-color, width, height',
                        'transition-duration': '0.3s'
                    }
                },
                {
                    selector: 'node[type="neighbor"]',
                    style: {
                        'background-color': '#a5a5a5',
                        'width': '25px',
                        'height': '25px',
                        'font-size': '8px'
                    }
                },
                {
                    selector: 'edge',
                    style: {
                        'curve-style': 'bezier',
                        'target-arrow-shape': 'triangle',
                        'opacity': 0.8,
                        'transition-property': 'line-color, width, opacity',
                        'transition-duration': '0.3s'
                    }
                },
                {
                    selector: ':selected',
                    style: {
                        'border-width': 3,
                        'border-color': '#764ba2',
                        'line-color': '#764ba2',
                        'target-arrow-color': '#764ba2',
                        'opacity': 1
                    }
                }
            ],
            layout: {
                name: 'cose',
                padding: 30
            }
        });

        // Add 3D View Button once
        if (!document.getElementById('view3dBtn')) {
            const btn = document.createElement('button');
            btn.id = 'view3dBtn';
            btn.className = 'btn btn-sm btn-outline-primary position-absolute';
            btn.style.top = '10px';
            btn.style.right = '10px';
            btn.style.zIndex = '1000';
            btn.innerHTML = '<i class="fas fa-cube me-1"></i> View 3D';
            btn.onclick = window.open3dGraph;
            cyContainer.appendChild(btn);
        }

        // --- Interaction Handlers ---

        // Hover effect for edges
        cy.on('mouseover', 'edge', function (evt) {
            const edge = evt.target;
            const data = edge.data();
            if (data.type === 'influence') return;

            edge.addClass('highlighted');

            if (tooltip) {
                tooltip.innerHTML = `
                    <div class="fw-bold text-primary mb-1">Interaction Path</div>
                    <div class="small mb-1">${edge.source().data('label')} + ${edge.target().data('label')}</div>
                    <div class="badge bg-${data.risk === 'HIGH' ? 'danger' : 'warning'} mb-2">${data.risk} Risk</div>
                    <div class="x-small text-muted">${data.mechanism}</div>
                `;
                tooltip.style.display = 'block';
            }
        });

        cy.on('tap', 'node', function (evt) {
            const node = evt.target;
            const label = node.data('label');
            const type = node.data('type');

            if (explanationCard && explanationText) {
                explanationCard.style.display = 'block';
                explanationText.innerHTML = `
                    <div class="mb-1"><strong>Drug: ${label}</strong></div>
                    <span class="badge ${type === 'target' ? 'bg-primary' : 'bg-secondary'} mb-2">${type.toUpperCase()}</span>
                    <p class="small text-muted">This drug is part of the local interaction neighborhood identified by the model.</p>
                `;
            }
        });
    }
}

// Open 3D Graph
window.open3dGraph = function () {
    window.open('/static/graph3d.html', '_blank');
}

// Check interactions
async function checkInteractions() {
    const resultsContainer = document.getElementById('resultsContainer');
    const checkBtn = document.getElementById('checkBtn');

    if (selectedDrugs.length < 2) return;

    // UI Loading State
    if (checkBtn) {
        checkBtn.disabled = true;
        checkBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> A.I. Analysis...';
    }
    if (resultsContainer) {
        resultsContainer.innerHTML = '<div class="text-center py-5"><div class="spinner-border text-primary"></div><p class="mt-2">Running Multi-Task GAT Model...</p></div>';
    }

    try {
        const drugIds = selectedDrugs.map(d => d.id);
        const response = await fetch(`${API_URL}/api/interactions/batch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ drugs: drugIds })
        });

        const results = await response.json();
        const interactions = results.interactions || results; // Handle both formats just in case
        displayResults(interactions);
        updateGraph(selectedDrugs, interactions); // Update graph visualization

    } catch (error) {
        console.error('Error:', error);
        if (resultsContainer) resultsContainer.innerHTML = '<div class="alert alert-danger">Error checking interactions. Ensure API is running.</div>';
    } finally {
        if (checkBtn) {
            checkBtn.disabled = false;
            checkBtn.innerHTML = '<i class="fas fa-search-plus"></i> Check Interactions';
        }
    }
}

function displayResults(results) {
    const container = document.getElementById('resultsContainer');
    if (!container) return;

    container.innerHTML = '';

    if (results.length === 0) {
        container.innerHTML = `
            <div class="alert alert-success">
                <i class="fas fa-check-circle me-2"></i>
                No clinically significant interactions found among these drugs.
            </div>
        `;
        return;
    }

    results.forEach((res, index) => {
        const isHighRisk = res.risk_level === 'HIGH' || res.severity === 'Major';
        const severityClass = res.severity ? `severity-${res.severity.toLowerCase()}` : '';
        const id = `explanation-${index}`;

        const card = document.createElement('div');
        card.className = `interaction-card ${isHighRisk ? 'border-danger' : 'border-warning'}`;
        card.style.borderLeft = isHighRisk ? '5px solid #dc3545' : '5px solid #ffc107';
        card.style.backgroundColor = 'white';
        card.style.padding = '15px';
        card.style.marginBottom = '15px';
        card.style.borderRadius = '10px';
        card.style.boxShadow = '0 2px 5px rgba(0,0,0,0.05)';

        // Format confidence
        const confScore = Math.round((res.confidence || res.probability) * 100);

        // Build Explanation Content
        let explanationHtml = '';
        if (res.clinical_explanation || res.model_explanation) {
            const clin = res.clinical_explanation || {};
            const mod = res.model_explanation || {};
            const feats = res.feature_contributions || {};
            const normalizeFeature = (value) => {
                const num = Number(value);
                if (!Number.isFinite(num)) return 0;
                return Math.min(1, Math.max(0, num));
            };
            const chemicalSimilarity = normalizeFeature(feats.chemical_similarity);
            const targetOverlap = normalizeFeature(feats.target_overlap);
            const atcSimilarity = normalizeFeature(feats.atc_similarity);

            explanationHtml = `
                <div id="${id}" class="explanation-panel">
                    <div class="row g-3">
                        <!-- Section 1: Risk Summary -->
                        <div class="col-12 mb-2 p-3 bg-light rounded-3 border">
                            <h6 class="fw-bold text-primary mb-1"><i class="fas fa-chart-line me-1"></i> 1. Risk Summary</h6>
                            <div class="d-flex justify-content-between align-items-center">
                                <p class="small mb-0"><strong>Calibrated Probability:</strong> ${(res.probability * 100).toFixed(1)}%</p>
                                <span class="badge bg-secondary">Calibrated</span>
                            </div>
                            <p class="small text-muted mt-1 mb-0">${clin.risk_summary || 'Analysis indicates potential for pharmacologic interference.'}</p>
                        </div>

                        <!-- Section 2: Clinical Mechanism -->
                        <div class="col-md-6">
                            <div class="explanation-header text-success"><i class="fas fa-stethoscope me-1"></i> 2. Clinical Mechanism</div>
                            <div class="p-2 border rounded bg-white">
                                <p class="small mb-1"><strong>Pathway:</strong> ${clin.mechanism || 'Pharmacologic interaction path'}</p>
                                <p class="small mb-1"><strong>Classification:</strong> ${clin.interaction_type || 'Combined'}</p>
                                <hr class="my-1">
                                <p class="small mb-0">${clin.clinical_reason || 'Monitored co-administration recommended.'}</p>
                            </div>
                        </div>

                        <!-- Section 3: Key Feature Similarity -->
                        <div class="col-md-6">
                            <div class="explanation-header text-info"><i class="fas fa-fingerprint me-1"></i> 3. Feature Breakdown</div>
                            <div class="p-2 border rounded bg-white">
                                <div class="mb-2">
                                    <div class="d-flex justify-content-between x-small mb-0">
                                        <span>Chemical Similarity</span>
                                        <span>${(chemicalSimilarity * 100).toFixed(0)}%</span>
                                    </div>
                                    <div class="progress" style="height: 6px;">
                                        <div class="progress-bar bg-info" style="width: ${chemicalSimilarity * 100}%"></div>
                                    </div>
                                </div>
                                <div class="mb-2">
                                    <div class="d-flex justify-content-between x-small mb-0">
                                        <span>Target/Pathway Overlap</span>
                                        <span>${(targetOverlap * 100).toFixed(0)}%</span>
                                    </div>
                                    <div class="progress" style="height: 6px;">
                                        <div class="progress-bar bg-primary" style="width: ${targetOverlap * 100}%"></div>
                                    </div>
                                </div>
                                <div class="mb-0">
                                    <div class="d-flex justify-content-between x-small mb-0">
                                        <span>ATC Class Matching</span>
                                        <span>${(atcSimilarity * 100).toFixed(0)}%</span>
                                    </div>
                                    <div class="progress" style="height: 6px;">
                                        <div class="progress-bar bg-success" style="width: ${atcSimilarity * 100}%"></div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Section 4: AI Graph Context -->
                        <div class="col-12">
                            <div class="explanation-header text-purple"><i class="fas fa-network-wired me-1"></i> 4. Graph Attention Analysis</div>
                            <div class="p-3 border rounded bg-white">
                                <p class="small font-italic mb-2">${mod.confidence_reasoning || 'Derived from neural graph embeddings.'}</p>
                                <div class="row">
                                    ${(mod.top_influencing_neighbors || []).slice(0, 4).map(n => `
                                        <div class="col-md-3 col-6 mb-2">
                                            <div class="p-2 border rounded text-center bg-light">
                                                <div class="small fw-bold text-truncate">${n.drug}</div>
                                                <div class="x-small text-muted">${n.relative_influence}% Influence</div>
                                            </div>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        </div>

                        <!-- Medical Disclaimer -->
                        <div class="col-12 mt-2">
                            <div class="p-2 bg-warning-subtle rounded border border-warning">
                                <p class="x-small text-dark mb-0">
                                    <i class="fas fa-exclamation-triangle me-1"></i> 
                                    <strong>Medical Disclaimer:</strong> ${clin.disclaimer || 'This AI prediction is for educational purposes and should not replace professional medical advice.'}
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        card.innerHTML = `
            <div class="d-flex justify-content-between align-items-start">
                <div>
                    <h5 class="mb-1">
                        <span class="fw-bold">${res.drug1.name}</span> + 
                        <span class="fw-bold">${res.drug2.name}</span>
                    </h5>
                    <div class="mb-2">
                        <span class="badge ${isHighRisk ? 'bg-danger' : 'bg-warning'} text-dark">
                            ${res.risk_level} RISK
                        </span>
                        ${res.severity ? `<span class="badge severity-badge ${severityClass} ms-2">${res.severity}</span>` : ''}
                        <span class="badge bg-info text-dark ms-2">Conf: ${confScore}%</span>
                        ${res.clinical_explanation || res.model_explanation ? `
                            <a class="why-link ms-3" onclick="toggleExplanation('${id}')">
                                <i class="fas fa-question-circle me-1"></i> Why?
                            </a>
                        ` : ''}
                    </div>
                    <div class="mt-2 p-2 bg-light rounded border-start border-4 border-info">
                        <strong><i class="fas fa-cogs"></i> Mechanism/Effect:</strong><br>
                        ${res.description}
                    </div>
                    <small class="text-muted mt-1 d-block"><i class="fas fa-robot me-1"></i> Source: ${res.source}</small>
                </div>
                <i class="fas ${isHighRisk ? 'fa-exclamation-triangle text-danger' : 'fa-exclamation-circle text-warning'} fa-2x"></i>
            </div>
            ${explanationHtml}
        `;
        container.appendChild(card);
    });
}

window.toggleExplanation = function (id) {
    const el = document.getElementById(id);
    if (!el) return;

    if (el.style.display === 'block') {
        el.style.display = 'none';
    } else {
        // Toggle all others off first for cleaner look? No, user might want to compare.
        el.style.display = 'block';
    }
}
