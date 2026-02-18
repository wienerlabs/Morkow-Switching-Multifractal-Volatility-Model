const CORTEX_CONFIG = {
    API_BASE: window.location.hostname === 'localhost'
        ? 'http://localhost:8000/api/v1'
        : '/api/v1',
    API_KEY: localStorage.getItem('cortex_api_key') || '',
    POLL_INTERVAL: 30000,
    TICKER_INTERVAL: 10000,
};
