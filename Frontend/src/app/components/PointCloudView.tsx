import { useEffect, useRef, useState } from 'react';

interface Point3D {
  x: number;
  y: number;
  z: number;
  color: string;
}

export function PointCloudView() {
  const topCanvasRef = useRef<HTMLCanvasElement>(null);
  const sideCanvasRef = useRef<HTMLCanvasElement>(null);
  const [points, setPoints] = useState<Point3D[]>([]);
  const [rowNow, setRowNow] = useState(10);
  const [trailRaw, setTrailRaw] = useState(239);

  // Generate random point cloud similar to the image
  useEffect(() => {
    const generatePoints = () => {
      const newPoints: Point3D[] = [];
      const colors = ['#FFFF00', '#00FFFF', '#FFD700', '#ADFF2F'];

      // Create central cluster (similar to image)
      const centerX = 150;
      const centerY = 150;
      const centerZ = 60;

      const numPoints = 8 + Math.floor(Math.random() * 5);
      for (let i = 0; i < numPoints; i++) {
        newPoints.push({
          x: centerX + (Math.random() - 0.5) * 25,
          y: centerY + (Math.random() - 0.5) * 35,
          z: centerZ + (Math.random() - 0.5) * 15,
          color: colors[Math.floor(Math.random() * colors.length)],
        });
      }

      // Add some scattered points
      for (let i = 0; i < 3; i++) {
        newPoints.push({
          x: 50 + Math.random() * 250,
          y: 50 + Math.random() * 250,
          z: 20 + Math.random() * 80,
          color: colors[Math.floor(Math.random() * colors.length)],
        });
      }

      setPoints(newPoints);
      setRowNow(10 + Math.floor(Math.random() * 3));
      setTrailRaw(238 + Math.floor(Math.random() * 4));
    };

    generatePoints();
    const interval = setInterval(generatePoints, 180);
    return () => clearInterval(interval);
  }, []);

  // Render Top XY view
  useEffect(() => {
    const canvas = topCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    // Dark purple/gray background like the image
    const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
    gradient.addColorStop(0, '#1a0f2e');
    gradient.addColorStop(1, '#2a1a3e');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid - denser like in the image
    ctx.strokeStyle = 'rgba(200, 200, 200, 0.25)';
    ctx.lineWidth = 0.8;
    const gridSize = 35;

    for (let x = 0; x <= canvas.width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y <= canvas.height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    // Axes labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.font = 'bold 16px monospace';
    ctx.fillText('Top XY', 20, 30);

    ctx.font = '14px monospace';
    ctx.fillText('X', canvas.width - 30, canvas.height - 15);
    ctx.fillText('Y', 15, 20);

    // Draw center axis lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1.5;
    // Vertical center line
    ctx.beginPath();
    ctx.moveTo(canvas.width / 2, 0);
    ctx.lineTo(canvas.width / 2, canvas.height);
    ctx.stroke();
    // Horizontal center line
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();

    // Draw points with glow effect
    points.forEach((point) => {
      const x = (point.x / 300) * canvas.width;
      const y = (point.y / 300) * canvas.height;

      // Glow effect
      ctx.shadowBlur = 10;
      ctx.shadowColor = point.color;

      ctx.fillStyle = point.color;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fill();

      ctx.shadowBlur = 0;
    });
  }, [points]);

  // Render Side YZ view
  useEffect(() => {
    const canvas = sideCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    // Dark purple/gray background
    const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
    gradient.addColorStop(0, '#1a0f2e');
    gradient.addColorStop(1, '#2a1a3e');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    ctx.strokeStyle = 'rgba(200, 200, 200, 0.25)';
    ctx.lineWidth = 0.8;
    const gridSize = 35;

    for (let x = 0; x <= canvas.width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y <= canvas.height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    // Axes labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.font = 'bold 16px monospace';
    ctx.fillText('Side YZ', 20, 30);

    ctx.font = '14px monospace';
    ctx.fillText('Y', canvas.width - 30, canvas.height - 15);
    ctx.fillText('Z', 15, 20);

    // Draw center axis lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1.5;
    // Vertical center line
    ctx.beginPath();
    ctx.moveTo(canvas.width / 2, 0);
    ctx.lineTo(canvas.width / 2, canvas.height);
    ctx.stroke();
    // Horizontal center line
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();

    // Draw points with glow effect
    points.forEach((point) => {
      const y = (point.y / 300) * canvas.width;
      const z = (point.z / 120) * canvas.height;

      // Glow effect
      ctx.shadowBlur = 10;
      ctx.shadowColor = point.color;

      ctx.fillStyle = point.color;
      ctx.beginPath();
      ctx.arc(y, z, 5, 0, Math.PI * 2);
      ctx.fill();

      ctx.shadowBlur = 0;
    });
  }, [points]);

  return (
    <div className="relative w-full h-full flex flex-col bg-[#0a0514]">
      {/* Title */}
      <div className="px-6 py-3 bg-gradient-to-r from-purple-900/30 to-transparent">
        <h2 className="text-white text-lg font-medium tracking-wide">mmWave Point Cloud (Top + Side)</h2>
      </div>

      {/* Top View - 2/3 height */}
      <div className="flex-[2] relative border-b border-white/10">
        <canvas ref={topCanvasRef} className="w-full h-full" />
      </div>

      {/* Side View - 1/3 height */}
      <div className="flex-1 relative border-b border-white/10">
        <canvas ref={sideCanvasRef} className="w-full h-full" />
      </div>

      {/* Bottom Stats */}
      <div className="px-6 py-2 bg-black/40 border-t border-white/20">
        <p className="text-white/90 text-sm font-mono tracking-wider">
          row_now: {rowNow} | trail_raw: {trailRaw} | cloud_used: {points.length}
        </p>
      </div>
    </div>
  );
}
