export const config = {
  api: {
    bodyParser: {
      sizeLimit: '15mb',
    },
  },
};

const HF_SPACE_URL = 'https://akashhhhwqx-ppe-safety-backend.hf.space';

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

    // 2. Call /reason endpoint
    const callHeaders = { 'Content-Type': 'application/json' };
    if (process.env.HF_TOKEN) {
      callHeaders['Authorization'] = `Bearer ${process.env.HF_TOKEN}`;
    }

    const callRes = await fetch(`${HF_SPACE_URL}/gradio_api/call/reason`, {
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
      const callErr = await callRes.text().catch(() => '');
      return res.status(callRes.status).json({
        error: `Gradio call failed (${callRes.status}): ${callErr.slice(0, 150)}`
      });
    }

    const { event_id } = await callRes.json();
    if (!event_id) {
      return res.status(502).json({ error: 'Gradio call returned no event_id' });
    }

    // 3. Read SSE stream
    const streamHeaders = {};
    if (process.env.HF_TOKEN) {
      streamHeaders['Authorization'] = `Bearer ${process.env.HF_TOKEN}`;
    }

    const streamRes = await fetch(`${HF_SPACE_URL}/gradio_api/call/reason/${event_id}`, {
      headers: streamHeaders
    });

    if (!streamRes.ok) {
      const streamErr = await streamRes.text().catch(() => '');
      return res.status(streamRes.status).json({
        error: `Gradio stream failed (${streamRes.status}): ${streamErr.slice(0, 150)}`
      });
    }

    const sseText = await streamRes.text();
    const lines = sseText.split('\n');

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const raw = JSON.parse(line.slice(6));
          const resultObj = typeof raw[0] === 'string' ? JSON.parse(raw[0]) : raw[0];
          if (resultObj) {
            return res.status(200).json(resultObj);
          }
        } catch (parseErr) {
          // continue checking
        }
      }
    }

    return res.status(200).json({ raw_sse: sseText });
  } catch (err) {
    return res.status(500).json({
      error: `Reasoning proxy error: ${err.message || 'Internal error'}`
    });
  }
}
