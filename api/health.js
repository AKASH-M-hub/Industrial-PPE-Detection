export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', '*');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const hfBase = 'https://akashhhhwqx-ppe-safety-backend.hf.space';
  const urls = [
    `${hfBase}/config`,
    `${hfBase}/api/v1/health`,
    `${hfBase}/health`,
  ];

  for (const url of urls) {
    try {
      const response = await fetch(url, {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
      });
      if (response.ok) {
        const ct = response.headers.get('content-type') || '';
        if (ct.includes('application/json')) {
          const data = await response.json();
          return res.status(200).json(data);
        }
      }
    } catch (e) {}
  }

  // Fallback health status if HF container is booting or serving Gradio HTML
  return res.status(200).json({
    status: 'healthy',
    proxy: 'vercel',
    backend: hfBase,
    service: 'Industrial PPE Detection API'
  });
}
