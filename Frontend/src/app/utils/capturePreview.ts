/** Capture a still frame from the Layer 8 MJPEG preview URL. */
export async function capturePreviewFrame(previewUrl: string): Promise<string | null> {
  const url = `${previewUrl}${previewUrl.includes('?') ? '&' : '?'}_snap=${Date.now()}`;
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      try {
        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth || img.width;
        canvas.height = img.naturalHeight || img.height;
        const ctx = canvas.getContext('2d');
        if (!ctx || canvas.width < 1 || canvas.height < 1) {
          resolve(null);
          return;
        }
        ctx.drawImage(img, 0, 0);
        resolve(canvas.toDataURL('image/jpeg', 0.9));
      } catch {
        resolve(null);
      }
    };
    img.onerror = () => resolve(null);
    img.src = url;
  });
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
