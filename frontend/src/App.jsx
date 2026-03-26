import React, { useState, useRef, useEffect } from 'react';
import { Terminal, UploadCloud, Activity, ShieldAlert, Cpu, Crosshair, Radar, AlertTriangle, CheckSquare, Satellite } from 'lucide-react';
import LiveTracker from './LiveTracker';

const TypewriterText = ({ text, delay = 50, onComplete }) => {
  const [displayedText, setDisplayedText] = useState('');
  
  useEffect(() => {
    let index = 0;
    setDisplayedText(''); // Reset on text change
    const interval = setInterval(() => {
      setDisplayedText(text.substring(0, index + 1));
      index++;
      if (index === text.length) {
        clearInterval(interval);
        if (onComplete) onComplete();
      }
    }, delay);
    return () => clearInterval(interval);
  }, [text, delay, onComplete]);

  return <span>{displayedText}</span>;
}

const ProgressBar = ({ progress, total = 20 }) => {
  const filled = Math.round((progress / 100) * total);
  const empty = total - filled;
  return (
    <span className="text-[#33ff00]">
      [{'|'.repeat(filled)}{'.'.repeat(empty)}] {Math.round(progress)}%
    </span>
  );
};

export default function App() {
  const [mode, setMode] = useState('manual'); // 'manual' | 'live'
  const [logs, setLogs] = useState([
    { id: 1, text: "INIT SYSTEM SECURE BOOT..." },
    { id: 2, text: "LOADING TACTICAL UPLINK PROTOCOLS..." },
    { id: 3, text: "RECONNAISSANCE MODULE ONLINE." }
  ]);
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [scanning, setScanning] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const logsEndRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const addLog = (text) => {
    setLogs(prev => [...prev, { id: Date.now(), text }]);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    if (!file.type.startsWith('image/')) {
      setError("INVALID FILE TYPE. ONLY IMAGES ACCEPTED.");
      addLog("ERROR: INVALID FILE UPLOADED. ABORTING.");
      return;
    }
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    setResults(null);
    setError(null);
    setScanProgress(0);
    addLog(`FILE ACCEPTED: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`);
    addLog(`AWAITING UPLINK COMMAND...`);
  };

  const triggerScan = async () => {
    if (!imageFile) {
      addLog("ERROR: NO IMAGE DETECTED IN DROPSHIP BUFFER.");
      return;
    }

    setScanning(true);
    setResults(null);
    setError(null);
    addLog("UPLINK INITIATED. TRANSMITTING IMAGE DATA...");

    // Simulated progress loop while waiting for response
    setScanProgress(0);
    const progressInterval = setInterval(() => {
      setScanProgress(p => (p < 90 ? p + Math.random() * 10 : p));
    }, 200);

    const formData = new FormData();
    formData.append('file', imageFile);

    try {
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${API_URL}/v1/detect`, {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);
      setScanProgress(100);

      if (!response.ok) {
        throw new Error(`SERVER_ERR CODE ${response.status}`);
      }

      const data = await response.json();
      setResults(data);
      addLog("TRANSMISSION SUCCESSFUL. DATA PARSED.");
      
      if (data.status === "IDENTIFIED") {
        addLog(`CRITICAL: TARGET ACQUIRED — ${data.target} (${(data.confidence * 100).toFixed(1)}%)`);
      } else {
        addLog(`STATUS: TARGET UNRESOLVED. CONFIDENCE TOO LOW.`);
      }

    } catch (err) {
      clearInterval(progressInterval);
      setScanProgress(0);
      setError(`CONNECTION FAILED: ${err.message}`);
      addLog(`CRITICAL ERROR: ${err.message}. AWAITING RETRY.`);
    } finally {
      setScanning(false);
    }
  };

  return (
    <div className="h-screen w-screen bg-[#0a0a0a] text-[#33ff00] font-mono flex flex-col">
      {/* Mode switcher */}
      <div className="flex border-b border-[#33ff00] shrink-0">
        <button
          onClick={() => setMode('manual')}
          className={`flex items-center gap-2 px-6 py-2 text-sm tracking-widest uppercase transition-all
            ${mode === 'manual' ? 'bg-[#33ff00] text-[#0a0a0a] font-bold' : 'text-[#33ff00] hover:bg-[#1a1a1a]'}`}
        >
          <Terminal size={14}/> Manual Scan
        </button>
        <button
          onClick={() => setMode('live')}
          className={`flex items-center gap-2 px-6 py-2 text-sm tracking-widest uppercase transition-all
            ${mode === 'live' ? 'bg-[#33ff00] text-[#0a0a0a] font-bold' : 'text-[#33ff00] hover:bg-[#1a1a1a]'}`}
        >
          <Satellite size={14}/> Live Satellite Feed
        </button>
      </div>

      {mode === 'live' ? <LiveTracker /> : (
    <div className="flex-1 p-4 flex flex-col gap-4 relative overflow-hidden">
      <div className="scanlines"></div>
      
      {/* HEADER */}
      <header className="flex justify-between items-center border border-[#33ff00] p-2 bg-[#0a0a0a] z-10">
        <div className="flex items-center gap-4">
          <Terminal size={24} className="text-[#33ff00]" />
          <h1 className="text-xl font-bold tracking-widest uppercase">
            <TypewriterText text="AEROSPACE DEFENSE COMMAND: ML TACTICAL UPLINK" delay={30} />
          </h1>
        </div>
        <div className="flex gap-4 text-sm">
          <span>SECURE_CONNECTION: ESTABLISHED</span>
          <span className="text-[#ffb000]">DEFCON: 2</span>
        </div>
      </header>

      {/* MAIN GRID */}
      <main className="flex-1 grid grid-cols-1 md:grid-cols-3 gap-4 min-h-0 z-10">
        
        {/* LEFT PANE: SYSTEM LOGS */}
        <section className="col-span-2 border border-[#33ff00] flex flex-col bg-[#0a0a0a]">
          <div className="border-b border-[#33ff00] p-1 bg-[#1a1a1a] flex justify-between items-center text-xs">
            <span>[ SYSTEM LOGS ]</span>
            <span>TMUX PANE 0</span>
          </div>
          <div className="flex-1 p-4 overflow-y-auto space-y-1">
            {logs.map((log) => (
              <div key={log.id} className="flex">
                <span className="mr-2">&gt;</span>
                {log.text}
              </div>
            ))}
            {scanning && (
              <div className="flex text-[#ffb000]">
                <span className="mr-2">&gt;</span>
                <span className="animate-pulse">SCANNING...</span> 
                <span className="ml-2"><ProgressBar progress={scanProgress} /></span>
              </div>
            )}
            <div ref={logsEndRef} />
            <div className="flex mt-2 items-center">
              <span className="text-[#33ff00] mr-2">user@recon:~$</span>
              <span className="w-2 h-4 bg-[#33ff00] animate-pulse inline-block"></span>
            </div>
          </div>
        </section>

        {/* RIGHT PANE: IMAGE UPLINK */}
        <section className="border border-[#33ff00] flex flex-col bg-[#0a0a0a]">
           <div className="border-b border-[#33ff00] p-1 bg-[#1a1a1a] flex justify-between items-center text-xs">
            <span>[ SENSOR UPLINK ]</span>
            <span>TMUX PANE 1</span>
          </div>
          <div className="flex-1 p-4 flex flex-col justify-center items-center gap-4">
            
            <div 
              className={`w-full h-48 border-2 border-dashed ${imageFile ? 'border-[#33ff00]' : 'border-neutral-700 hover:border-[#33ff00] cursor-pointer'} flex flex-col items-center justify-center relative overflow-hidden transition-colors`}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                className="hidden" 
                accept="image/*" 
                onChange={handleFileInput}
              />
              {imagePreview ? (
                <>
                  <img src={imagePreview} alt="Target radar view" className="absolute inset-0 w-full h-full object-cover opacity-60 filter brightness-110 contrast-150 grayscale" style={{mixBlendMode: 'screen'}} />
                  <div className="absolute inset-0 bg-[#33ff00] opacity-10"></div>
                  {/* Fake CRT scanning line overlay */}
                  <div className="absolute top-0 left-0 w-full h-2 bg-[#33ff00] opacity-40 shadow-[0_0_10px_#33ff00] animate-[scan_3s_linear_infinite]"></div>
                  <style>{`@keyframes scan { 0% { top: -10px; } 100% { top: 100%; } }`}</style>
                </>
              ) : (
                <>
                  <UploadCloud size={48} className="mb-2 text-neutral-500" />
                  <span className="text-sm text-center px-4">DRAG & DROP IMAGERY OR CLICK TO BROWSE</span>
                  <span className="text-xs text-neutral-500 mt-2">SUPPORTED FORMATS: JPG, PNG, WEBP</span>
                </>
              )}
            </div>

            <button 
              onClick={triggerScan}
              disabled={!imageFile || scanning}
              className={`w-full border p-2 font-bold tracking-widest flex items-center justify-center gap-2 uppercase transition-all
                ${!imageFile ? 'border-neutral-700 text-neutral-700 cursor-not-allowed' : 
                  scanning ? 'border-[#ffb000] text-[#ffb000] cursor-wait' : 
                  'border-[#33ff00] text-[#0a0a0a] bg-[#33ff00] hover:bg-[#0a0a0a] shadow-[0_0_10px_#33ff00] hover:text-[#33ff00]'
                }`}
            >
              {scanning ? (
                <><Activity className="animate-spin" size={20}/> PROCESSING...</>
              ) : (
                <><Crosshair size={20}/> EXECUTE SCAN</>
              )}
            </button>
            {error && <div className="text-red-500 text-xs border border-red-500 p-2 w-full truncate">{error}</div>}
          </div>
        </section>

      </main>

      {/* BOTTOM PANE: TARGET ANALYSIS */}
      <section className="h-1/3 border border-[#33ff00] flex flex-col bg-[#0a0a0a] z-10 shrink-0">
        <div className="border-b border-[#33ff00] p-1 bg-[#1a1a1a] flex justify-between items-center text-xs">
          <span>[ TARGET ANALYSIS & TELEMETRY ]</span>
          <span>TMUX PANE 2</span>
        </div>
        <div className="p-4 overflow-y-auto w-full h-full flex flex-col">
          {scanning ? (
             <div className="flex-1 flex flex-col justify-center items-center opacity-70">
                <Radar size={48} className="animate-spin text-[#ffb000] mb-4" />
                <div className="text-[#ffb000]">ANALYZING FRAME... <ProgressBar progress={scanProgress} total={30} /></div>
             </div>
          ) : results ? (
            <div className="w-full text-sm">
              <div className="mb-2 text-[#ffb000]">
                {">>"} PARSING RESPONSE TIMESTAMP: {new Date().toISOString()}
              </div>
              <table className="w-full border-collapse">
                <thead>
                  <tr className="border-b border-[#33ff00] text-left">
                    <th className="p-2 w-16">ID</th>
                    <th className="p-2">CLASSIFICATION</th>
                    <th className="p-2 w-32">CONFIDENCE</th>
                    <th className="p-2">LATENCY</th>
                    <th className="p-2">THREAT LEVEL</th>
                  </tr>
                </thead>
                <tbody>
                  {results.target ? (
                    <tr className="border-b border-dashed border-neutral-700 hover:bg-[#1a1a1a]">
                      <td className="p-2 text-center">#1</td>
                      <td className="p-2 font-bold">{results.target}</td>
                      <td className="p-2">
                        <div className="flex items-center gap-2">
                          <span>{(results.confidence * 100).toFixed(1)}%</span>
                          <div className="h-2 w-16 bg-[#0a0a0a] border border-[#33ff00]">
                            <div className="h-full bg-[#33ff00]" style={{width: `${results.confidence * 100}%`}}></div>
                          </div>
                        </div>
                      </td>
                      <td className="p-2 text-neutral-400 text-xs">
                        {results.telemetry?.latency_ms} ms
                      </td>
                      <td className="p-2">
                         {results.confidence > 0.8 ? (
                          <span className="text-red-500 font-bold flex items-center gap-1"><ShieldAlert size={14}/> HIGH</span>
                         ) : results.confidence > 0.5 ? (
                          <span className="text-[#ffb000] font-bold flex items-center gap-1"><AlertTriangle size={14}/> MED</span>
                         ) : (
                          <span className="text-green-500 font-bold flex items-center gap-1"><CheckSquare size={14}/> LOW</span>
                         )}
                      </td>
                    </tr>
                  ) : (
                    <tr>
                      <td colSpan="5" className="p-8 text-center text-neutral-500">
                        NO TARGETS DETECTED. SECTOR CLEAR.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>

              {/* RAW JSON Output as fallback/extra telemetry */}
              <div className="mt-4 mt-8 border-t border-dashed border-[#33ff00] pt-4">
                 <div className="mb-2 text-[#33ff00] font-bold text-xs uppercase">{">>"} RAW_DATALINK_DUMP</div>
                 <pre className="text-xs text-neutral-500 max-h-32 overflow-y-auto p-2 bg-[#050505] border border-neutral-800">
                   {JSON.stringify(results, null, 2)}
                 </pre>
              </div>

            </div>
          ) : (
            <div className="flex-1 flex flex-col justify-center items-center text-neutral-600">
               <Cpu size={48} className="mb-4 text-neutral-800"/>
               <span>SYSTEM IDLE. AWAITING UPLINK PACKETS.</span>
            </div>
          )}
        </div>
      </section>
    </div>
      )}
    </div>
  );
}
