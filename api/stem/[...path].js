const GPU_SERVER_URL = "https://cinecut-gpu-v42.loca.lt";

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', '*');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  try {
    const { path } = req.query;
    const pathStr = Array.isArray(path) ? path.join('/') : path;
    const targetUrl = `${GPU_SERVER_URL}/api/stem/${pathStr}`;

    const gpuResponse = await fetch(targetUrl, {
      headers: {
        'bypass-tunnel-reminder': 'true',
        'Bypass-Tunnel-Reminder': 'true'
      }
    });

    const contentType = gpuResponse.headers.get('content-type') || 'audio/wav';
    const arrayBuffer = await gpuResponse.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    res.setHeader('Content-Type', contentType);
    return res.status(200).send(buffer);
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
}
