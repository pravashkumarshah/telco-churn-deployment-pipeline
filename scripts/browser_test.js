const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

const CHROME_PATH = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const REPORTS_DIR = path.join(__dirname, '..', 'reports', 'browser');

async function runBrowserTest() {
  if (!fs.existsSync(REPORTS_DIR)) {
    fs.mkdirSync(REPORTS_DIR, { recursive: true });
  }

  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu']
  });

  const page = await browser.newPage();
  const consoleLogs = [];
  const networkRequests = [];

  page.on('console', msg => {
    consoleLogs.push({ type: msg.type(), text: msg.text() });
  });

  page.on('response', async res => {
    const url = res.url();
    if (url.includes('/health') || url.includes('/predict') || url.includes('/drift')) {
      let body = '';
      try { body = await res.json(); } catch(e) { body = await res.text(); }
      networkRequests.push({
        url: url,
        status: res.status(),
        statusText: res.statusText(),
        body: body
      });
    }
  });

  console.log('--- 1. Testing Desktop Viewport (1440x900) ---');
  await page.setViewport({ width: 1440, height: 900 });
  await page.goto('http://127.0.0.1:8000/', { waitUntil: 'networkidle0' });

  const title = await page.title();
  console.log('Page Title:', title);

  // Set API Key via UI
  console.log('--- 2. Setting API Key in Browser Modal ---');
  await page.evaluate(() => {
    openKeyModal();
    const input = document.getElementById('modal-key-input');
    input.value = 'telco-secret-key-churn-2026-secure';
    saveKey();
  });
  await new Promise(r => setTimeout(r, 600));

  const keyStatus = await page.evaluate(() => {
    return {
      activeKeySet: !!activeKey,
      label: document.getElementById('key-btn-label')?.textContent,
      dotClass: document.getElementById('key-status-dot')?.className
    };
  });
  console.log('Key Status in UI:', keyStatus);

  // Test Single Prediction
  console.log('--- 3. Testing Single Prediction (Studio) ---');
  await page.evaluate(() => {
    applyPreset('high_risk');
    executeStudioScore();
  });
  await new Promise(r => setTimeout(r, 1000));

  const studioResults = await page.evaluate(() => {
    return {
      scoreText: document.getElementById('result-score-val')?.textContent,
      statusTag: document.getElementById('result-status-tag')?.textContent,
      threshold: document.getElementById('result-threshold')?.textContent,
      directiveTitle: document.getElementById('directive-title')?.textContent,
      pinPosition: {
        left: document.getElementById('curve-point-pin')?.style.left,
        top: document.getElementById('curve-point-pin')?.style.top
      },
      pinTagText: document.getElementById('pin-tag-text')?.textContent
    };
  });
  console.log('Studio Results in UI:', studioResults);
  await page.screenshot({ path: path.join(REPORTS_DIR, '01_desktop_studio.png'), fullPage: false });

  // Test Batch Prediction
  console.log('--- 4. Testing Batch Prediction (Cohort Matrix) ---');
  await page.evaluate(() => {
    setTab('batch');
    loadDemoBatch(10);
  });
  await new Promise(r => setTimeout(r, 400));
  await page.evaluate(() => executeBatch());
  await new Promise(r => setTimeout(r, 1500));

  const batchResults = await page.evaluate(() => {
    const rows = document.querySelectorAll('#matrix-table-body tr');
    return {
      rowCount: rows.length,
      firstRow: rows[0] ? rows[0].innerText.replace(/\n/g, ' | ') : null,
      summaryTotal: document.getElementById('batch-metric-total')?.textContent,
      summaryHighRisk: document.getElementById('batch-metric-churn')?.textContent,
      summaryAvgRisk: document.getElementById('batch-metric-avg')?.textContent
    };
  });
  console.log('Batch Results in UI:', batchResults);
  await page.screenshot({ path: path.join(REPORTS_DIR, '02_desktop_batch.png'), fullPage: false });

  // Test Drift Scan
  console.log('--- 5. Testing Drift Scan Diagnostics ---');
  await page.evaluate(() => {
    setTab('drift');
  });
  await new Promise(r => setTimeout(r, 400));
  await page.evaluate(() => triggerDriftScan());
  await new Promise(r => setTimeout(r, 1500));

  const driftResults = await page.evaluate(() => {
    const rows = document.querySelectorAll('#drift-tbody tr');
    return {
      rowCount: rows.length,
      firstFeature: rows[0] ? rows[0].innerText.replace(/\n/g, ' | ') : null,
      statusTitle: document.getElementById('drift-status-title')?.textContent
    };
  });
  console.log('Drift Results in UI:', driftResults);
  await page.screenshot({ path: path.join(REPORTS_DIR, '03_desktop_drift.png'), fullPage: false });

  // Test Mobile Responsive Viewport (375x812)
  console.log('--- 6. Testing Mobile Responsive Viewport (375x812) ---');
  await page.setViewport({ width: 375, height: 812, isMobile: true, hasTouch: true });
  await page.evaluate(() => setTab('studio'));
  await new Promise(r => setTimeout(r, 600));

  const mobileMetrics = await page.evaluate(() => {
    return {
      bodyScrollWidth: document.body.scrollWidth,
      windowWidth: window.innerWidth,
      hasHorizontalOverflow: document.body.scrollWidth > window.innerWidth,
      headerVisible: !!document.getElementById('main-header'),
      introVisible: !document.getElementById('intro-cards-section').classList.contains('hidden')
    };
  });
  console.log('Mobile Metrics:', mobileMetrics);
  await page.screenshot({ path: path.join(REPORTS_DIR, '04_mobile_studio.png'), fullPage: false });

  await browser.close();

  const report = {
    test_timestamp: new Date().toISOString(),
    page_title: title,
    desktop_tests: {
      key_status: keyStatus,
      single_prediction: studioResults,
      batch_prediction: batchResults,
      drift_scan: driftResults
    },
    mobile_tests: mobileMetrics,
    network_requests: networkRequests,
    console_logs: consoleLogs
  };

  fs.writeFileSync(path.join(REPORTS_DIR, 'browser_audit_results.json'), JSON.stringify(report, null, 2), 'utf8');
  console.log('--- Browser Audit Successfully Completed and Saved ---');
}

runBrowserTest().catch(err => {
  console.error('Browser Test Error:', err);
  process.exit(1);
});
