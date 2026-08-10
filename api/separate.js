const GPU_SERVER_URL = "https://replication-gives-mambo-gig.trycloudflare.com";

export const config = {
  api: {
    bodyParser: false,
  },
};

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', '*');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  try {
    const targetUrl = `${GPU_SERVER_URL}/api/separate-audio`;
    
    const chunks = [];
    for await (const chunk of req) {
      chunks.push(chunk);
    }
    const bodyBuffer = Buffer.concat(chunks);

    const gpuResponse = await fetch(targetUrl, {
      method: req.method,
      headers: {
        'content-type': req.headers['content-type'] || 'multipart/form-data',
        'bypass-tunnel-reminder': 'true',
        'Bypass-Tunnel-Reminder': 'true'
      },
      body: bodyBuffer
    });

    const contentType = gpuResponse.headers.get('content-type') || '';

    if (contentType.includes('json')) {
      const data = await gpuResponse.json();
      return res.status(200).json(data);
    } else {
      const arrayBuffer = await gpuResponse.arrayBuffer();
      const buffer = Buffer.from(arrayBuffer);
      res.setHeader('Content-Type', contentType || 'audio/wav');
      return res.status(200).send(buffer);
    }
  } catch (error) {
    console.error("Vercel GPU proxy error:", error);
    return res.status(500).json({ error: error.message });
  }
}
