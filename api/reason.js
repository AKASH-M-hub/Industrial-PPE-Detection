export const config = {
  api: {
    bodyParser: {
      sizeLimit: '15mb',
    },
  },
};

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', '*');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ detail: 'Method not allowed. Use POST.' });
  }

  const hfBase = 'https://akashhhhwqx-ppe-safety-backend.hf.space';
  const targetEndpoints = [
    `${hfBase}/api/v1/reason`,
    `${hfBase}/reason`,
  ];

  let bodyData;
  if (typeof req.body === 'string') {
    bodyData = req.body;
  } else if (req.body && typeof req.body === 'object') {
    bodyData = JSON.stringify(req.body);
  } else {
    bodyData = '';
  }

  let lastRes = null;
  let lastErr = null;

  for (const url of targetEndpoints) {
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: bodyData,
      });

      if (response.ok) {
        const json = await response.json();
        return res.status(200).json(json);
      }

      lastRes = response;
      if (response.status === 404 || response.status === 405) {
        continue;
      }

      const errText = await response.text();
      return res.status(response.status).send(errText);
    } catch (err) {
      lastErr = err;
    }
  }

  if (lastRes) {
    const errText = await lastRes.text();
    return res.status(lastRes.status).send(errText);
  }

  return res.status(502).json({
    detail: `Vercel proxy failed to reach reasoning backend: ${lastErr ? lastErr.message : 'Unknown error'}`
  });
}
