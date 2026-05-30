const SAMPLE_RATE = 16000;
const SEND_FRAME_SECONDS = 0.2;
const WORD_APPEND_DELAY_MS = 45;

const els = {
  healthPill: document.querySelector("#healthPill"),
  modelSelect: document.querySelector("#modelSelect"),
  includeMicCheckbox: document.querySelector("#includeMicCheckbox"),
  chunkSelect: document.querySelector("#chunkSelect"),
  languageSelect: document.querySelector("#languageSelect"),
  startButton: document.querySelector("#startButton"),
  stopButton: document.querySelector("#stopButton"),
  clearButton: document.querySelector("#clearButton"),
  copyButton: document.querySelector("#copyButton"),
  statusText: document.querySelector("#statusText"),
  timerText: document.querySelector("#timerText"),
  chunkText: document.querySelector("#chunkText"),
  transcriptOutput: document.querySelector("#transcriptOutput"),
  eventLog: document.querySelector("#eventLog"),
};

const state = {
  websocket: null,
  streams: [],
  audioContext: null,
  sourceNodes: [],
  outputNode: null,
  processorNode: null,
  startedAt: 0,
  timerId: 0,
  chunksSent: 0,
  pendingBuffers: [],
  pendingLength: 0,
  transcript: [],
  transcriptParagraphs: [[]],
  interimTranscript: null,
  wordQueue: [],
  wordTimer: 0,
  isRunning: false,
  isStopping: false,
  stopCloseTimer: 0,
  health: null,
};

function setStatus(text) {
  els.statusText.textContent = text;
}

function languageLabel() {
  return els.languageSelect.value === "es" ? "Spanish" : "English";
}

function updateHealthPill() {
  if (!state.health) {
    els.healthPill.textContent = `Backend · ${languageLabel()}`;
    return;
  }
  if (state.health.device === "cuda" && state.health.cuda_ready === false) {
    els.healthPill.textContent = `CUDA missing libs · ${languageLabel()}`;
    return;
  }
  els.healthPill.textContent = `${state.health.device} / ${state.health.compute_type} · ${languageLabel()}`;
}

function logEvent(text) {
  if (!els.eventLog) {
    return;
  }
  const row = document.createElement("div");
  row.className = "log-row";
  row.textContent = text;
  els.eventLog.prepend(row);
  while (els.eventLog.children.length > 9) {
    els.eventLog.lastElementChild.remove();
  }
}

function describeMediaError(error) {
  const name = error?.name || "";
  const message = error?.message || String(error);
  if (name === "NotAllowedError") {
    return "The browser blocked or canceled capture. Open the app in a real Chrome/Chromium tab, not an IDE preview.";
  }
  if (name === "InvalidStateError") {
    return "Capture must start from an active tab. Click the page and press Transcribe again.";
  }
  if (name === "NotFoundError") {
    return "No screen capture source is available.";
  }
  if (name === "NotReadableError") {
    return "The system did not allow screen capture. Check browser or desktop permissions.";
  }
  return message;
}

function formatClock(seconds) {
  const mins = Math.floor(seconds / 60).toString().padStart(2, "0");
  const secs = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${mins}:${secs}`;
}

function updateRunningUi(running) {
  state.isRunning = running;
  els.startButton.disabled = running;
  els.stopButton.disabled = !running;
  els.modelSelect.disabled = running;
  els.includeMicCheckbox.disabled = running;
  els.chunkSelect.disabled = running;
  els.languageSelect.disabled = running;
  document.body.classList.toggle("is-recording", running);
}

function updateTimer() {
  if (!state.startedAt) {
    els.timerText.textContent = "00:00";
    return;
  }
  const seconds = (Date.now() - state.startedAt) / 1000;
  els.timerText.textContent = formatClock(seconds);
}

function clearRunTimers() {
  window.clearInterval(state.timerId);
  state.timerId = 0;
  if (state.stopCloseTimer) {
    window.clearTimeout(state.stopCloseTimer);
    state.stopCloseTimer = 0;
  }
}

async function releaseAudioCapture() {
  if (state.processorNode) {
    state.processorNode.disconnect();
    state.processorNode.onaudioprocess = null;
  }
  for (const node of state.sourceNodes) {
    node.disconnect();
  }
  if (state.outputNode) {
    state.outputNode.disconnect();
  }
  if (state.audioContext) {
    await state.audioContext.close().catch(() => {});
  }
  for (const stream of state.streams) {
    for (const track of stream.getTracks()) {
      track.stop();
    }
  }

  state.processorNode = null;
  state.sourceNodes = [];
  state.outputNode = null;
  state.audioContext = null;
  state.streams = [];
}

function finalizeStop(status = "Stopped") {
  clearRunTimers();
  state.startedAt = 0;
  state.pendingBuffers = [];
  state.pendingLength = 0;
  state.interimTranscript = null;
  state.isStopping = false;
  updateRunningUi(false);
  setStatus(status);
  renderTranscript();
  updateTimer();
}

function websocketUrl(model, language) {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const params = new URLSearchParams({
    model,
    language,
    source: "browser",
    chunk_seconds: els.chunkSelect.value,
  });
  return `${protocol}//${window.location.host}/ws/transcribe?${params.toString()}`;
}

function renderTranscript() {
  els.transcriptOutput.innerHTML = "";
  const hasWords = state.transcriptParagraphs.some((paragraph) => paragraph.length > 0);
  const interimWords = state.interimTranscript?.text?.trim().split(/\s+/).filter(Boolean) || [];
  if (!hasWords && state.wordQueue.length === 0 && interimWords.length === 0) {
    const placeholder = document.createElement("p");
    placeholder.className = "placeholder";
    placeholder.textContent = "Press Transcribe and share tab audio. Text will appear word by word.";
    els.transcriptOutput.append(placeholder);
    return;
  }

  const visibleParagraphs = state.transcriptParagraphs.filter(
    (paragraph, index) => paragraph.length > 0 || index === state.transcriptParagraphs.length - 1,
  );
  visibleParagraphs.forEach((paragraph, index) => {
    if (paragraph.length === 0 && index < visibleParagraphs.length - 1) {
      return;
    }
    const flow = document.createElement("p");
    flow.className = "transcript-flow";
    for (const word of paragraph) {
      const span = document.createElement("span");
      span.className = "transcript-word";
      span.textContent = word;
      flow.append(span, document.createTextNode(" "));
    }
    if (index === visibleParagraphs.length - 1) {
      const cursor = document.createElement("span");
      cursor.className = "transcript-cursor";
      cursor.setAttribute("aria-hidden", "true");
      flow.append(cursor);
    }
    els.transcriptOutput.append(flow);
  });

  if (interimWords.length > 0) {
    const flow = document.createElement("p");
    flow.className = "transcript-flow transcript-interim";
    for (const word of interimWords) {
      const span = document.createElement("span");
      span.className = "transcript-word";
      span.textContent = word;
      flow.append(span, document.createTextNode(" "));
    }
    const cursor = document.createElement("span");
    cursor.className = "transcript-cursor";
    cursor.setAttribute("aria-hidden", "true");
    flow.append(cursor);
    els.transcriptOutput.append(flow);
  }
  els.transcriptOutput.scrollTop = els.transcriptOutput.scrollHeight;
}

function ensureCurrentParagraph() {
  if (state.transcriptParagraphs.length === 0) {
    state.transcriptParagraphs.push([]);
  }
}

function drainWordQueue() {
  if (state.wordQueue.length === 0) {
    state.wordTimer = 0;
    renderTranscript();
    return;
  }

  const nextItem = state.wordQueue.shift();
  if (nextItem?.type === "paragraph") {
    const current = state.transcriptParagraphs[state.transcriptParagraphs.length - 1];
    if (current && current.length > 0) {
      state.transcriptParagraphs.push([]);
    }
  } else if (nextItem?.type === "word") {
    ensureCurrentParagraph();
    state.transcriptParagraphs[state.transcriptParagraphs.length - 1].push(nextItem.value);
    renderTranscript();
  }
  state.wordTimer = window.setTimeout(drainWordQueue, WORD_APPEND_DELAY_MS);
}

function enqueueTranscriptText(text, paragraphBreakBefore = false) {
  const words = text.trim().split(/\s+/).filter(Boolean);
  if (words.length === 0) {
    return;
  }
  const hasWords = state.transcriptParagraphs.some((paragraph) => paragraph.length > 0);
  if (paragraphBreakBefore && hasWords) {
    state.wordQueue.push({ type: "paragraph" });
  }
  state.wordQueue.push(...words.map((word) => ({ type: "word", value: word })));
  if (!state.wordTimer) {
    drainWordQueue();
  }
}

function clearTranscript() {
  if (state.wordTimer) {
    window.clearTimeout(state.wordTimer);
    state.wordTimer = 0;
  }
  state.transcript = [];
  state.transcriptParagraphs = [[]];
  state.interimTranscript = null;
  state.wordQueue = [];
  state.chunksSent = 0;
  els.chunkText.textContent = "0";
  renderTranscript();
  logEvent("Transcription cleared");
}

function downsample(input, inputRate, outputRate) {
  if (inputRate === outputRate) {
    return new Float32Array(input);
  }
  const ratio = inputRate / outputRate;
  const outputLength = Math.max(1, Math.round(input.length / ratio));
  const output = new Float32Array(outputLength);
  let inputOffset = 0;

  for (let outputOffset = 0; outputOffset < outputLength; outputOffset += 1) {
    const nextInputOffset = Math.min(input.length, Math.round((outputOffset + 1) * ratio));
    let sum = 0;
    let count = 0;
    for (let i = inputOffset; i < nextInputOffset; i += 1) {
      sum += input[i];
      count += 1;
    }
    output[outputOffset] = count > 0 ? sum / count : 0;
    inputOffset = nextInputOffset;
  }
  return output;
}

function floatToPcm16(input) {
  const output = new Int16Array(input.length);
  for (let i = 0; i < input.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, input[i]));
    output[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return output.buffer;
}

function pushAudio(buffer) {
  const targetLength = Math.round(SEND_FRAME_SECONDS * SAMPLE_RATE);
  state.pendingBuffers.push(buffer);
  state.pendingLength += buffer.length;

  while (state.pendingLength >= targetLength) {
    const chunk = new Float32Array(targetLength);
    let offset = 0;

    while (offset < targetLength && state.pendingBuffers.length > 0) {
      const head = state.pendingBuffers[0];
      const needed = targetLength - offset;
      if (head.length <= needed) {
        chunk.set(head, offset);
        offset += head.length;
        state.pendingBuffers.shift();
      } else {
        chunk.set(head.subarray(0, needed), offset);
        state.pendingBuffers[0] = head.subarray(needed);
        offset += needed;
      }
    }

    state.pendingLength -= targetLength;
    if (state.websocket?.readyState === WebSocket.OPEN) {
      state.websocket.send(floatToPcm16(chunk));
    }
  }
}

async function fetchModels() {
  const response = await fetch("/api/models");
  if (!response.ok) {
    throw new Error("Could not load model list.");
  }
  const payload = await response.json();
  els.modelSelect.innerHTML = "";

  for (const model of payload.models) {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = model.recommended
      ? `${model.label} - recommended`
      : model.label;
    option.title = model.detail;
    els.modelSelect.append(option);
  }

  const defaultModel = payload.default_model || "tiny";
  if ([...els.modelSelect.options].some((option) => option.value === defaultModel)) {
    els.modelSelect.value = defaultModel;
  } else {
    els.modelSelect.value = "tiny";
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) {
      throw new Error("Backend unavailable.");
    }
    const payload = await response.json();
    state.health = payload;
    updateHealthPill();
    els.healthPill.classList.toggle("is-ok", payload.cuda_ready !== false);
    els.healthPill.classList.toggle("is-error", payload.cuda_ready === false);
    if (payload.cuda_ready === false) {
      logEvent(`Missing CUDA libraries: ${payload.cuda_missing_libraries.join(", ")}`);
    }
  } catch (error) {
    els.healthPill.textContent = "Backend offline";
    els.healthPill.classList.add("is-error");
  }
}

function stopStreams(streams) {
  for (const stream of streams) {
    for (const track of stream.getTracks()) {
      track.stop();
    }
  }
}

function watchCaptureTracks(streams) {
  for (const stream of streams) {
    for (const track of stream.getTracks()) {
      track.addEventListener("ended", () => {
        if (!state.isRunning || state.isStopping) {
          return;
        }
        const message = "Capture ended";
        setStatus(message);
        void stop().then(() => setStatus(message));
      });
    }
  }
}

async function openCaptureStreams(includeMic) {
  if (!window.isSecureContext) {
    throw new Error("Screen capture requires opening the app at http://127.0.0.1:8099 or HTTPS.");
  }
  if (!navigator.mediaDevices?.getDisplayMedia) {
    throw new Error("This browser does not support screen capture. Try Chrome/Chromium at http://127.0.0.1:8099.");
  }

  const streams = [];
  const displayStream = await navigator.mediaDevices.getDisplayMedia({
    video: true,
    audio: true,
  });
  streams.push(displayStream);

  if (displayStream.getAudioTracks().length === 0) {
    stopStreams(streams);
    throw new Error("No audio was shared. Enable audio sharing in the browser picker.");
  }

  if (includeMic) {
    if (!navigator.mediaDevices?.getUserMedia) {
      stopStreams(streams);
      throw new Error("This browser does not support microphone capture.");
    }
    try {
      const micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      });
      streams.push(micStream);
    } catch (error) {
      stopStreams(streams);
      throw error;
    }
  }

  return streams;
}

function attachAudioProcessor(streams) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    throw new Error("AudioContext is not available in this browser.");
  }

  const audioContext = new AudioContextClass();
  const processorNode = audioContext.createScriptProcessor(4096, 1, 1);
  const sourceNodes = [];

  processorNode.onaudioprocess = (event) => {
    const output = event.outputBuffer.getChannelData(0);
    output.fill(0);
    if (!state.isRunning) {
      return;
    }
    const input = event.inputBuffer.getChannelData(0);
    const normalized = downsample(input, audioContext.sampleRate, SAMPLE_RATE);
    pushAudio(normalized);
  };

  for (const stream of streams) {
    const sourceNode = audioContext.createMediaStreamSource(stream);
    const gainNode = audioContext.createGain();
    gainNode.gain.value = 1;
    sourceNode.connect(gainNode);
    gainNode.connect(processorNode);
    sourceNodes.push(sourceNode, gainNode);
  }
  processorNode.connect(audioContext.destination);

  state.audioContext = audioContext;
  state.sourceNodes = sourceNodes;
  state.outputNode = null;
  state.processorNode = processorNode;
}

function attachSocket(model, language) {
  const socket = new WebSocket(websocketUrl(model, language));
  state.websocket = socket;

  socket.addEventListener("open", () => {
    setStatus("Connected");
    logEvent("WebSocket connected");
  });

  socket.addEventListener("message", (event) => {
    let payload;
    try {
      payload = JSON.parse(event.data);
    } catch {
      return;
    }

    if (Number.isFinite(payload.chunk)) {
      state.chunksSent = Math.max(state.chunksSent, payload.chunk);
      els.chunkText.textContent = String(state.chunksSent);
    }

    if (payload.type === "ready") {
      setStatus("Listening");
      logEvent(payload.message);
      return;
    }
    if (payload.type === "stopped") {
      const socket = state.websocket;
      state.websocket = null;
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.close();
      }
      finalizeStop();
      return;
    }
    if (payload.type === "status" || payload.type === "processing") {
      setStatus(payload.message || "Processing");
      return;
    }
    if (payload.type === "silence") {
      setStatus("Silence");
      return;
    }
    if (payload.type === "empty_phrase") {
      if (state.interimTranscript?.phrase === payload.phrase) {
        state.interimTranscript = null;
        renderTranscript();
      }
      setStatus("Listening");
      return;
    }
    if (payload.type === "interim_transcript") {
      if (payload.text) {
        state.interimTranscript = {
          phrase: payload.phrase,
          text: payload.text,
          paragraphBreakBefore: Boolean(payload.paragraph_break_before),
        };
        renderTranscript();
      }
      setStatus("Listening");
      return;
    }
    if (payload.type === "transcript") {
      if (payload.text) {
        if (state.interimTranscript?.phrase === payload.phrase) {
          state.interimTranscript = null;
        }
        state.transcript.push({
          start: payload.start || 0,
          end: payload.end || 0,
          text: payload.text,
        });
        enqueueTranscriptText(payload.text, Boolean(payload.paragraph_break_before));
        setStatus("Listening");
      }
      return;
    }
    if (payload.type === "error") {
      const message = payload.message || "Transcription error";
      setStatus(message);
      logEvent(message);
      void stop().then(() => setStatus(message));
    }
  });

  socket.addEventListener("close", () => {
    if (state.isStopping) {
      state.websocket = null;
      finalizeStop();
      return;
    }
    if (state.isRunning) {
      state.websocket = null;
      finalizeStop("Connection closed");
    }
  });

  socket.addEventListener("error", () => {
    const message = "Could not connect to backend";
    setStatus(message);
    logEvent(message);
  });
}

async function start() {
  try {
    const model = els.modelSelect.value;
    if (!model) {
      throw new Error("No model selected. Check that the backend is running.");
    }

    const includeMic = els.includeMicCheckbox.checked;
    setStatus(includeMic ? "Opening screen and microphone" : "Opening screen picker");
    const streams = await openCaptureStreams(includeMic);

    clearTranscript();
    updateRunningUi(true);
    setStatus("Connecting");
    state.startedAt = Date.now();
    state.timerId = window.setInterval(updateTimer, 500);
    updateTimer();

    state.streams = streams;
    watchCaptureTracks(state.streams);
    attachSocket(model, els.languageSelect.value);
    attachAudioProcessor(state.streams);
  } catch (error) {
    const message = describeMediaError(error);
    logEvent(message);
    await stop();
    setStatus(message);
  }
}

async function stop() {
  if (state.isStopping) {
    return;
  }

  state.isStopping = true;
  updateRunningUi(false);
  setStatus("Stopping");
  window.clearInterval(state.timerId);
  state.pendingBuffers = [];
  state.pendingLength = 0;
  await releaseAudioCapture();

  const socket = state.websocket;
  if (!socket || socket.readyState === WebSocket.CLOSED || socket.readyState === WebSocket.CLOSING) {
    state.websocket = null;
    finalizeStop();
    return;
  }

  if (socket.readyState === WebSocket.OPEN) {
    try {
      socket.send(JSON.stringify({ type: "stop" }));
    } catch {
      socket.close();
    }
  } else if (socket.readyState === WebSocket.CONNECTING) {
    socket.close();
  }

  state.stopCloseTimer = window.setTimeout(() => {
    const activeSocket = state.websocket;
    if (activeSocket && activeSocket.readyState !== WebSocket.CLOSED) {
      activeSocket.close();
    }
    state.websocket = null;
    finalizeStop();
  }, 1200);
}

async function copyTranscript() {
  const text = state.transcript.map((item) => item.text).join("\n").trim();
  if (!text) {
    return;
  }
  await navigator.clipboard.writeText(text);
  logEvent("Text copied");
}

els.startButton.addEventListener("click", start);
els.stopButton.addEventListener("click", stop);
els.clearButton.addEventListener("click", clearTranscript);
els.copyButton.addEventListener("click", copyTranscript);
els.languageSelect.addEventListener("change", updateHealthPill);

els.languageSelect.value = "en";
updateHealthPill();

try {
  await Promise.all([fetchModels(), checkHealth()]);
} catch (error) {
  const message = error.message || String(error);
  setStatus(message);
  logEvent(message);
}
renderTranscript();
