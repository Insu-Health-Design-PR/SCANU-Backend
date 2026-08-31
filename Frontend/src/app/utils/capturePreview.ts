/** Map UI camera sensor → Layer 8 screenshot IPC sensor id. */
function screenshotSensor(sensor: 'webcam' | 'multi_camera' | 'thermal' | 'mmwave'): string {
  if (sensor === 'multi_camera') return 'multi_camera';
  if (sensor === 'thermal') return 'thermal';
  return 'webcam';
}

/** Capture one still via IPC snapshot (does not open an MJPEG stream). */
export async function capturePreviewFrame(
  previewUrl: string,
  sensor: 'webcam' | 'multi_camera' | 'thermal' | 'mmwave' = 'webcam',
): Promise<string | null> {
  const snapPath = `/api/screenshot/${screenshotSensor(sensor)}/frame?_=${Date.now()}`;
  try {
    const res = await fetch(snapPath, { cache: 'no-store' });
    if (!res.ok) return null;
    const blob = await res.blob();
    if (!blob.size) return null;
    return await new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(typeof reader.result === 'string' ? reader.result : null);
      reader.onerror = () => resolve(null);
      reader.readAsDataURL(blob);
    });
  } catch {
    // Legacy fallback: only if snapshot route unavailable (old backend).
    void previewUrl;
    return null;
  }
}

export function downloadDataUrl(dataUrl: string, filename: string) {
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = filename;
  a.click();
}

export function slabScreenshotName(slabId: string, sensor: string) {
  const ts = new Date().toISOString().replace(/[:.]/g, '-');
  return `${slabId}_${sensor}_${ts}.jpg`;
}
