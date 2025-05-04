// DOM elements
const searchInput = document.getElementById('search-input');
const searchButton = document.getElementById('search-button');
const loadingContainer = document.getElementById('loading-container');
const resultsContainer = document.getElementById('results-container');
const searchInfo = document.getElementById('search-info');
const searchStats = document.getElementById('search-stats');
const searchMode = document.getElementById('search-mode');
const enhancedQuery = document.getElementById('enhanced-query');
const noResults = document.getElementById('no-results');
const exampleQueries = document.querySelectorAll('.example-query');

// Options elements
const enhanceQueryOption = document.getElementById('enhanceQuery');
const explainResultsOption = document.getElementById('explainResults');

// Initialize popovers and tooltips
function initBootstrapComponents() {
    const popoverTriggerList = document.querySelectorAll('[data-bs-toggle="popover"]');
    [...popoverTriggerList].map(popoverTriggerEl => new bootstrap.Popover(popoverTriggerEl));
    
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
}

// Format a score as a percentage with color coding
function formatScore(score) {
    const percentage = Math.round(score * 100);
    let colorClass = '';
    
    if (percentage >= 90) {
        colorClass = 'bg-success';
    } else if (percentage >= 75) {
        colorClass = 'bg-primary';
    } else if (percentage >= 60) {
        colorClass = 'bg-info';
    } else if (percentage >= 45) {
        colorClass = 'bg-warning';
    } else {
        colorClass = 'bg-secondary';
    }
    
    return `<span class="badge ${colorClass}">${percentage}%</span>`;
}

// Create card HTML for a place
function createPlaceCard(place) {
    // Format image section
    let imageSection = '';
    if (place.image_urls && place.image_urls.length > 0) {
        imageSection = `
            <img src="${place.image_urls[0]}" class="card-img-top" alt="${place.name}" 
                 onerror="this.parentNode.innerHTML='<div class=\\'image-placeholder\\'><i class=\\'bi bi-image fs-1\\'></i></div>'">
            <span class="score-badge">${Math.round(place.score * 100)}% match</span>
        `;
    } else if (place.image_url) {
        imageSection = `
            <img src="${place.image_url}" class="card-img-top" alt="${place.name}" 
                 onerror="this.parentNode.innerHTML='<div class=\\'image-placeholder\\'><i class=\\'bi bi-image fs-1\\'></i></div>'">
            <span class="score-badge">${Math.round(place.score * 100)}% match</span>
        `;
    } else {
        imageSection = `
            <div class="image-placeholder">
                <i class="bi bi-image fs-1"></i>
            </div>
            <span class="score-badge">${Math.round(place.score * 100)}% match</span>
        `;
    }
    
    // Format tags section
    const tagsSection = place.vibe_tags && place.vibe_tags.length > 0 
        ? place.vibe_tags.map(tag => `<span class="badge vibe-tag">${tag.replace(/_/g, ' ')}</span>`).join('')
        : '';
    
    // Format explanation section
    const explanationSection = place.match_reason 
        ? `<div class="explanation"><i class="bi bi-lightbulb me-1"></i> ${place.match_reason}</div>`
        : '';
    
    // Build complete card HTML
    return `
        <div class="card h-100">
            ${imageSection}
            <div class="card-body">
                <h5 class="card-title">${place.emoji || '📍'} ${place.name}</h5>
                <div class="d-flex align-items-center mb-2">
                    <i class="bi bi-geo-alt me-1"></i>
                    <span class="location-badge">${place.neighborhood || 'New York'}</span>
                </div>
                <p class="card-text small">${place.short_description || ''}</p>
                ${explanationSection}
                <div class="d-flex flex-wrap mb-2">
                    ${tagsSection}
                </div>
            </div>
            <div class="card-footer d-flex justify-content-between align-items-center">
                <a href="https://maps.google.com/?q=${encodeURIComponent(place.name + ', ' + (place.neighborhood || 'New York'))}" 
                   class="btn btn-sm btn-outline-primary" target="_blank">
                    <i class="bi bi-map"></i> Maps
                </a>
                <button class="btn btn-sm btn-outline-secondary"
                        data-bs-toggle="popover"
                        data-bs-placement="top"
                        data-bs-html="true"
                        data-bs-title="${place.name}"
                        data-bs-content="<div><strong>Location:</strong> ${place.neighborhood || 'New York'}</div><div><strong>Score:</strong> ${Math.round(place.score * 100)}%</div>">
                    <i class="bi bi-info-circle"></i>
                </button>
            </div>
        </div>
    `;
}

// Perform search
async function performSearch() {
    const query = searchInput.value.trim();
    if (!query) return;
    
    // Show loading state
    loadingContainer.style.display = 'block';
    resultsContainer.innerHTML = '';
    searchInfo.classList.add('d-none');
    enhancedQuery.classList.add('d-none');
    noResults.classList.add('d-none');
    
    try {
        // Get options
        const enhance = enhanceQueryOption.checked;
        const explain = explainResultsOption.checked;
        
        // Call the search API
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                limit: 20,
                use_text: true,
                use_images: true,
                enhance: enhance,
                explain: explain,
                rerank: true
            })
        });
        
        const data = await response.json();
        
        // Check for errors
        if (!response.ok) {
            throw new Error(data.detail || 'Search failed');
        }
        
        const results = data.results || [];
        
        // Display search info
        searchStats.textContent = `Found ${results.length} places in ${data.processing_time.toFixed(2)} seconds`;
        
        // Display search mode
        searchMode.textContent = `Search modes: Text=${data.text_search_used}, Image=${data.image_search_used}`;
        
        // Show search info section
        searchInfo.classList.remove('d-none');
        
        // Display enhanced query if available
        if (data.processed_query && data.processed_query !== data.original_query) {
            enhancedQuery.textContent = `Enhanced query: ${data.processed_query}`;
            enhancedQuery.classList.remove('d-none');
        }
        
        // Show no results message if needed
        if (results.length === 0) {
            noResults.classList.remove('d-none');
            return;
        }
        
        // Display neighborhoods if available
        if (data.neighborhoods && data.neighborhoods.length > 0) {
            const neighborhoodText = document.createElement('div');
            neighborhoodText.className = 'mt-2';
            neighborhoodText.textContent = `Detected neighborhoods: ${data.neighborhoods.join(', ')}`;
            searchStats.appendChild(neighborhoodText);
        }
        
        // Display results
        results.forEach(place => {
            const colElement = document.createElement('div');
            colElement.className = 'col';
            colElement.innerHTML = createPlaceCard(place);
            resultsContainer.appendChild(colElement);
        });
        
        // Initialize Bootstrap components
        initBootstrapComponents();
        
    } catch (error) {
        console.error('Search error:', error);
        resultsContainer.innerHTML = `
            <div class="col-12">
                <div class="alert alert-danger">
                    <i class="bi bi-exclamation-triangle me-2"></i>
                    Error: ${error.message || 'Failed to perform search'}
                </div>
            </div>
        `;
    } finally {
        // Hide loading state
        loadingContainer.style.display = 'none';
    }
}

// Event listeners
searchButton.addEventListener('click', performSearch);

searchInput.addEventListener('keyup', (event) => {
    if (event.key === 'Enter') {
        performSearch();
    }
});

// Example query click handlers
exampleQueries.forEach(example => {
    example.addEventListener('click', () => {
        searchInput.value = example.textContent;
        performSearch();
    });
});

// Initialize the page
document.addEventListener('DOMContentLoaded', () => {
    // Check for URL parameters for initial search
    const urlParams = new URLSearchParams(window.location.search);
    const initialQuery = urlParams.get('q');
    if (initialQuery) {
        searchInput.value = initialQuery;
        performSearch();
    }
    
    // Initialize Bootstrap components
    initBootstrapComponents();
});