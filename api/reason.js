export const config = {
  api: {
    bodyParser: {
      sizeLimit: '15mb',
    },
  },
};

const HF_SPACE_URL = 'https://akashhhhwqx-ppe-safety-backend.hf.space';

async function executeGradioReason(endpoint, filePath, question) {
  const callHeaders = { 'Content-Type': 'application/json' };
  if (process.env.HF_TOKEN) {
    callHeaders['Authorization'] = `Bearer ${process.env.HF_TOKEN}`;
  }

  const callRes = await fetch(`${HF_SPACE_URL}/gradio_api/call/${endpoint}`, {
    method: 'POST',
    headers: callHeaders,
    body: JSON.stringify({
      data: [
        { path: filePath, meta: { _type: 'gradio.FileData' } },
        question
      ]
    }),
  });

  if (!callRes.ok) {
    return { success: false, error: `Queue status ${callRes.status}` };
  }

  const { event_id } = await callRes.json();
  if (!event_id) {
    return { success: false, error: 'No event_id returned' };
  }

  const streamHeaders = {};
  if (process.env.HF_TOKEN) {
    streamHeaders['Authorization'] = `Bearer ${process.env.HF_TOKEN}`;
  }

  const streamRes = await fetch(`${HF_SPACE_URL}/gradio_api/call/${endpoint}/${event_id}`, {
    headers: streamHeaders
  });

  if (!streamRes.ok) {
    return { success: false, error: `Stream status ${streamRes.status}` };
  }

  const sseText = await streamRes.text();
  const lines = sseText.split('\n');

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      try {
        const raw = JSON.parse(line.slice(6));
        if (Array.isArray(raw)) {
          const resultObj = typeof raw[0] === 'string' ? JSON.parse(raw[0]) : raw[0];
          if (resultObj && !resultObj.error) {
            return { success: true, data: resultObj };
          }
          if (resultObj && resultObj.error) {
            return { success: false, error: resultObj.error };
          }
        } else if (raw && raw.error) {
          const isQuota = String(raw.error).toLowerCase().includes('quota') || String(raw.error).toLowerCase().includes('limit');
          return { success: false, is_quota: isQuota, error: raw.error };
        }
      } catch (pe) {}
    }
  }

  return { success: false, error: 'Empty result from stream', raw: sseText };
}

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

  try {
    let imgBuffer = null;
    let question = 'Is he wearing helmet or not?';

    if (req.body && typeof req.body === 'object') {
      const rawImage = req.body.image || req.body.file;
      if (rawImage && typeof rawImage === 'string') {
        const cleanB64 = rawImage.includes(',') ? rawImage.split(',')[1] : rawImage;
        imgBuffer = Buffer.from(cleanB64, 'base64');
      }
      if (req.body.question) {
        question = req.body.question;
      }
    } else if (typeof req.body === 'string' && req.body.length > 0) {
      try {
        const parsed = JSON.parse(req.body);
        const rawImage = parsed.image || parsed.file;
        if (rawImage) {
          const cleanB64 = rawImage.includes(',') ? rawImage.split(',')[1] : rawImage;
          imgBuffer = Buffer.from(cleanB64, 'base64');
        }
        if (parsed.question) question = parsed.question;
      } catch (e) {
        imgBuffer = Buffer.from(req.body, 'base64');
      }
    } else if (Buffer.isBuffer(req.body)) {
      imgBuffer = req.body;
    }

    if (!imgBuffer || imgBuffer.length === 0) {
      return res.status(400).json({ error: 'No image data provided in request.' });
    }

    // 1. Upload to Gradio
    const boundary = '----WebKitFormBoundary' + Math.random().toString(36).substring(2);
    const header = Buffer.from(`--${boundary}\r\nContent-Disposition: form-data; name="files"; filename="image.jpg"\r\nContent-Type: image/jpeg\r\n\r\n`);
    const footer = Buffer.from(`\r\n--${boundary}--\r\n`);
    const payload = Buffer.concat([header, imgBuffer, footer]);

    const uploadHeaders = {
      'Content-Type': `multipart/form-data; boundary=${boundary}`,
    };
    if (process.env.HF_TOKEN) {
      uploadHeaders['Authorization'] = `Bearer ${process.env.HF_TOKEN}`;
    }

    const uploadRes = await fetch(`${HF_SPACE_URL}/gradio_api/upload`, {
      method: 'POST',
      headers: uploadHeaders,
      body: payload,
    });

    if (!uploadRes.ok) {
      const uploadErr = await uploadRes.text().catch(() => '');
      return res.status(uploadRes.status).json({
        error: `Gradio upload failed (${uploadRes.status}): ${uploadErr.slice(0, 150)}`
      });
    }

    const files = await uploadRes.json();
    const filePath = files && files[0];
    if (!filePath) {
      return res.status(502).json({ error: 'Gradio upload returned empty file path' });
    }

    // 2. Try ZeroGPU reasoning first
    let result = await executeGradioReason('reason', filePath, question);

    // 3. If ZeroGPU quota is exceeded or GPU fails, fallback to CPU reasoning
    if (!result.success) {
      console.warn(`ZeroGPU reason notice: ${result.error}. Attempting CPU fallback...`);
      result = await executeGradioReason('reason_cpu', filePath, question);
    }

    if (result.success && result.data) {
      return res.status(200).json(result.data);
    }

    return res.status(502).json({
      error: `Reasoning failed on both GPU and CPU: ${result.error || 'Unknown error'}`
    });
  } catch (err) {
    return res.status(500).json({
      error: `Reasoning proxy error: ${err.message || 'Internal error'}`
    });
  }
}
