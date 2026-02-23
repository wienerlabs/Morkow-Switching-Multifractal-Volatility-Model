const CORTEX_CONFIG = {
    API_BASE: window.location.hostname === 'localhost'
        ? 'http://localhost:8000/api/v1'
        : '/api/v1',
    API_KEY: localStorage.getItem('cortex_api_key') || '',
    POLL_INTERVAL: 30000,
    TICKER_INTERVAL: 10000,
    STAKING_PROGRAM_ID: 'rYantWFyB4PsL36r9XB7nUb8TQ1pAhn9A87S6TbpMsr',
    VESTING_PROGRAM_ID: '5PDicSrsh9zyVMwDjL61WXHuNkzQTk6rpCs5CnGzpXns',
    CRTX_MINT: 'HAUqFj3uYsFt6PhMgztaVkRi5RC3mFWxyLceJzCRDevg',
};
