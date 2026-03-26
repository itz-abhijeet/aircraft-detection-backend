import { useEffect, useState, useRef } from 'react';
import { ref, query, orderByChild, limitToLast, onValue } from 'firebase/database';
import { db } from './firebase';
import { Satellite, Radar, ShieldAlert, AlertTriangle, CheckSquare, Cpu, Activity } from 'lucide-react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

const planeIcon = new L.DivIcon({
  html: `<div style="width:14px;height:14px;background:#33ff00;border:2px solid #0a0a0a;box-shadow:0 0 10px #33ff00,0 0 20px #33ff00;transform:rotate(45deg);"></div>`,
  iconSize: [14, 14], iconAnchor: [7, 7], className: '',
});

const INDIA_CENTER = [22.5, 80.0];
const INDIA_ZOOM = 5;
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function MapRecenter({ lat, lon }) {
  const map = useMap();
  useEffect(() => { if (lat && lon) map.flyTo([lat, lon], 8, { duration: 1.5 }); }, [lat, lon]);
  return null;
}

const ProgressBar = ({ progress, total = 20 }) => {
  const filled = Math.round((progress / 100) * total);
  return <span>[{'|'.repeat(filled)}{'.'.repeat(total - filled)}] {Math.round(progress)}%</span>;
};

export default function LiveTracker() {
  const [logs, setLogs] = useState([
    { id: 1, text: "SATELLITE UPLINK INITIALIZING..." },
    { id: 2, text: "CONNECTING TO FIREBASE REALTIME DATABASE..." },
    { id: 3, text: "MONITORING plane_trackings NODE..." },
  ]);
  const [connected, setConnected] = useState(false);
  const [tracking, setTracking] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);

  const [scanning, setScanning] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const logsEndRef = useRef(null);
  const lastKeyRef = useRef(null);

  useEffect(() => { logsEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [logs]);
  const addLog = (text) => setLogs(prev => [...prev, { id: Date.now(), text }]);

  useEffect(() => {
    const q = query(ref(db, 'plane_trackings'), orderByChild('timestamp'), limitToLast(1));
    const unsub = onValue(q, (snapshot) => {
      if (!snapshot.exists()) return;
      setConnected(true);
      snapshot.forEach((child) => {
        if (child.key === lastKeyRef.current) return;
        lastKeyRef.current = child.key;
        const data = child.val();
        setTracking({ key: child.key, ...data });
        setImagePreview(data.imageUrl || null);
        setResult(null); setError(null);
        addLog(`NEW TRACKING: ${child.key}`);
        addLog(`POS: LAT ${data.estimatedPlaneLat?.toFixed(5)}, LON ${data.estimatedPlaneLon?.toFixed(5)}`);
        addLog(`HDG: ${data.heading?.toFixed(1)}° — INITIATING ML SCAN...`);
        classifyTracking(data.imageUrl);
      });
    });
    return () => unsub();
  }, []);

  const classifyTracking = async (imageUrl) => {
    if (!imageUrl) { addLog("ERROR: NO IMAGE URL."); return; }
    setScanning(true); setScanProgress(0);
    const iv = setInterval(() => setScanProgress(p => p < 90 ? p + Math.random() * 10 : p), 200);
    try {
      const res = await fetch(`${API_URL}/v1/detect-url`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_url: imageUrl }),
      });
      clearInterval(iv); setScanProgress(100);
      if (!res.ok) throw new Error(`SERVER_ERR ${res.status}`);
      const data = await res.json();
      setResult(data);
      addLog(`SCAN COMPLETE: ${data.target} — ${(data.confidence * 100).toFixed(1)}%`);
      addLog(`STATUS: ${data.status} | LATENCY: ${data.telemetry?.latency_ms}ms`);
    } catch (err) {
      clearInterval(iv); setScanProgress(0);
      setError(`SCAN FAILED: ${err.message}`);
      addLog(`ERROR: ${err.message}`);
    } finally { setScanning(false); }
  };


  return (
    <div className="flex-1 p-4 flex flex-col gap-4 relative overflow-hidden">
      <div className="scanlines"></div>
      <header className="shrink-0 flex justify-between items-center border border-[#33ff00] p-2 bg-[#0a0a0a] z-10">
        <div className="flex items-center gap-4">
          <Satellite size={24} />
          <h1 className="text-xl font-bold tracking-widest uppercase">SATELLITE RECON FEED: LIVE TARGET ACQUISITION</h1>
        </div>
        <div className="flex gap-4 text-sm">
          <span className={connected ? 'text-[#33ff00]' : 'text-neutral-600'}>FIREBASE: {connected ? 'CONNECTED' : 'CONNECTING...'}</span>
          <span className="text-[#ffb000]">AUTO-SCAN: ACTIVE</span>
        </div>
      </header>

      <main className="flex-1 grid gap-4 min-h-0 z-10" style={{ gridTemplateColumns: '2fr 1fr' }}>
        <section className="border border-[#33ff00] flex flex-col min-h-0">
          <div className="shrink-0 border-b border-[#33ff00] p-1 bg-[#1a1a1a] flex justify-between items-center text-xs">
            <span>[ TACTICAL MAP — AIRSPACE MONITOR ]</span><span>TMUX PANE 0</span>
          </div>
          <div className="flex-1 min-h-0">
            <MapContainer center={INDIA_CENTER} zoom={INDIA_ZOOM} style={{ height: '100%', width: '100%' }} zoomControl={true}>
              <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" attribution="&copy; CARTO" />
              {tracking?.estimatedPlaneLat && (
                <>
                  <Marker position={[tracking.estimatedPlaneLat, tracking.estimatedPlaneLon]} icon={planeIcon}>
                    <Popup>
                      <div style={{ fontFamily: 'monospace', fontSize: '11px', background: '#111', color: '#33ff00', padding: '6px' }}>
                        <div style={{ fontWeight: 'bold' }}>{result?.target || 'SCANNING...'}</div>
                        <div>LAT: {tracking.estimatedPlaneLat?.toFixed(5)}</div>
                        <div>LON: {tracking.estimatedPlaneLon?.toFixed(5)}</div>
                        <div>HDG: {tracking.heading?.toFixed(1)}°</div>
                        {result && <div>CONF: {(result.confidence * 100).toFixed(1)}%</div>}
                      </div>
                    </Popup>
                  </Marker>
                  <MapRecenter lat={tracking.estimatedPlaneLat} lon={tracking.estimatedPlaneLon} />
                </>
              )}
            </MapContainer>
          </div>
        </section>

        <section className="flex flex-col gap-4 min-h-0">
          <div className="shrink-0 border border-[#33ff00] flex flex-col bg-[#0a0a0a]">
            <div className="shrink-0 border-b border-[#33ff00] p-1 bg-[#1a1a1a] flex justify-between items-center text-xs">
              <span>[ SATELLITE IMAGERY ]</span><span>TMUX PANE 1</span>
            </div>
            <div className="relative bg-black flex items-center justify-center overflow-hidden" style={{ height: '220px' }}>
              {imagePreview ? (
                <>
                  <img src={imagePreview} alt="Target" className="w-full h-full object-contain" style={{ filter: 'brightness(1.15) contrast(1.2)' }} />
                  <div className="absolute top-0 left-0 w-full h-1 bg-[#33ff00] opacity-60 animate-[scan_3s_linear_infinite]" />
                  <style>{`@keyframes scan{0%{top:-4px}100%{top:100%}}`}</style>
                </>
              ) : (
                <div className="flex flex-col items-center text-neutral-700 gap-2">
                  <Satellite size={36} /><span className="text-xs">AWAITING IMAGERY</span>
                </div>
              )}
              {scanning && <div className="absolute inset-0 bg-black/50 flex items-center justify-center"><Activity className="animate-spin text-[#ffb000]" size={28} /></div>}
            </div>
          </div>

          <div className="flex-1 min-h-0 border border-[#33ff00] flex flex-col bg-[#0a0a0a]">
            <div className="shrink-0 border-b border-[#33ff00] p-1 bg-[#1a1a1a] flex justify-between items-center text-xs">
              <span>[ FEED LOGS ]</span><span>TMUX PANE 2</span>
            </div>
            <div className="flex-1 min-h-0 p-3 overflow-y-auto space-y-1 text-xs">
              {logs.map(l => (<div key={l.id} className="flex"><span className="mr-2 shrink-0">&gt;</span><span>{l.text}</span></div>))}
              {scanning && <div className="flex text-[#ffb000]"><span className="mr-2">&gt;</span><span className="animate-pulse">SCANNING... </span><span className="ml-1"><ProgressBar progress={scanProgress} total={15} /></span></div>}
              {error && <div className="text-red-500 border border-red-500 p-1 mt-1">{error}</div>}
              <div ref={logsEndRef} />
              <div className="flex mt-1 items-center"><span className="mr-2">sat@recon:~$</span><span className="w-2 h-3 bg-[#33ff00] animate-pulse inline-block"></span></div>
            </div>
          </div>
        </section>
      </main>


      <section className="shrink-0 border border-[#33ff00] flex flex-col bg-[#0a0a0a] z-10" style={{ height: '155px' }}>
        <div className="shrink-0 border-b border-[#33ff00] p-1 bg-[#1a1a1a] flex justify-between items-center text-xs">
          <span>[ TARGET ANALYSIS & TELEMETRY ]</span><span>TMUX PANE 3</span>
        </div>
        <div className="flex-1 min-h-0 p-3 overflow-y-auto">
          {scanning ? (
            <div className="flex items-center gap-4 h-full">
              <Radar size={32} className="animate-spin text-[#ffb000] shrink-0" />
              <span className="text-[#ffb000] text-sm">ANALYZING... <ProgressBar progress={scanProgress} total={30} /></span>
            </div>
          ) : result && tracking ? (
            <div className="text-sm">
              <div className="mb-2 text-[#ffb000] text-xs">{">>"} UPLINK: {new Date(tracking.timestamp).toISOString()}</div>
              <table className="w-full border-collapse text-xs">
                <thead>
                  <tr className="border-b border-[#33ff00] text-left">
                    <th className="p-1 pr-4">CLASSIFICATION</th><th className="p-1 pr-4">CONFIDENCE</th>
                    <th className="p-1 pr-4">LATITUDE</th><th className="p-1 pr-4">LONGITUDE</th>
                    <th className="p-1 pr-4">HEADING</th><th className="p-1">THREAT</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="hover:bg-[#1a1a1a]">
                    <td className="p-1 pr-4 font-bold text-[#33ff00]">{result.target}</td>
                    <td className="p-1 pr-4">
                      <div className="flex items-center gap-2">
                        <span>{(result.confidence * 100).toFixed(1)}%</span>
                        <div className="h-2 w-16 bg-[#0a0a0a] border border-[#33ff00]">
                          <div className="h-full bg-[#33ff00]" style={{ width: `${result.confidence * 100}%` }} />
                        </div>
                      </div>
                    </td>
                    <td className="p-1 pr-4 text-neutral-300">{tracking.estimatedPlaneLat?.toFixed(6)}</td>
                    <td className="p-1 pr-4 text-neutral-300">{tracking.estimatedPlaneLon?.toFixed(6)}</td>
                    <td className="p-1 pr-4 text-neutral-300">{tracking.heading?.toFixed(1)}°</td>
                    <td className="p-1">
                      {result.confidence > 0.8
                        ? <span className="text-red-500 font-bold flex items-center gap-1"><ShieldAlert size={12}/> HIGH</span>
                        : result.confidence > 0.5
                        ? <span className="text-[#ffb000] font-bold flex items-center gap-1"><AlertTriangle size={12}/> MED</span>
                        : <span className="text-green-500 font-bold flex items-center gap-1"><CheckSquare size={12}/> LOW</span>}
                    </td>
                  </tr>
                </tbody>
              </table>
              <div className="mt-2 flex gap-6 text-xs text-neutral-500">
                <span>STATUS: <span className={result.status === 'IDENTIFIED' ? 'text-[#33ff00]' : 'text-[#ffb000]'}>{result.status}</span></span>
                <span>LATENCY: {result.telemetry?.latency_ms}ms</span>
                <span>MODEL: {result.telemetry?.model}</span>
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-4 h-full text-neutral-600">
              <Cpu size={32} className="text-neutral-800 shrink-0" />
              <span className="text-sm">SYSTEM IDLE. AWAITING SATELLITE UPLINK PACKETS.</span>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
