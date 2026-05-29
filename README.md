# Transcript Direct

Webapp local para transcribir audio en vivo con `faster-whisper`. No tiene login:
abres la pagina, eliges modelo/idioma y presionas `Transcribir`.

La app captura audio de una pantalla o pestana desde el navegador. Opcionalmente
puede mezclar el microfono del navegador en la misma senal antes de enviarla a
Whisper. La salida se agrupa de forma natural: el backend emite una frase cuando
detecta una pausa y usa `Frase max.` solo como limite de seguridad.

## Requisitos

- Linux desktop.
- Python 3.10 o superior.
- Chrome/Chromium para captura de pestana/pantalla con audio.
- GPU CUDA recomendada para `large-v3`; CPU funciona, pero sera mucho mas lento.
- Node.js solo es necesario para el chequeo rapido de sintaxis del frontend.

## Instalacion

```bash
git clone <repo-url> transcript-direct
cd transcript-direct
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si ya tienes el entorno de `PUDU_app`, tambien puedes usarlo sin crear `.venv`:

```bash
PYTHON_BIN=../PUDU_app/backend/.venv/bin/python ./run-webapp.sh
```

## Ejecutar

```bash
./run-webapp.sh
```

Abre la app en una pestana real del navegador:

```text
http://127.0.0.1:8099
```

Usa `127.0.0.1` o `localhost`. La captura de pantalla/pestana del navegador no
funciona de forma confiable en previews internos de IDE ni en hosts remotos sin
HTTPS.

Si el puerto esta ocupado:

```bash
PORT=8100 ./run-webapp.sh
```

## Uso

Defaults recomendados:

- Modelo: `Whisper large-v3`.
- Fuente: pantalla o pestana con audio desde el navegador.
- Idioma: `Ingles`.
- Frase max.: `3 s`.
- Contexto entre frases: `24` palabras.

Flujo normal:

1. Abre `http://127.0.0.1:8099`.
2. Selecciona modelo e idioma.
3. Activa `Incluir microfono del navegador` solo si quieres mezclar tu voz con
   el audio de la pestana/pantalla.
4. Presiona `Transcribir`.
5. En el selector del navegador, elige pestana/pantalla y marca compartir audio.

## Modelos

La carpeta `models/` existe en el repo solo como estructura:

```text
models/
  .gitkeep
  whisper-cache/
    .gitkeep
```

El contenido real de modelos esta ignorado por Git. En el primer uso,
`faster-whisper` puede descargar modelos en `models/whisper-cache/`. Si existe
`../PUDU_app/backend/models/whisper`, la app tambien lista esos modelos locales.

Variables utiles:

```bash
WHISPER_MODEL_NAME=large-v3 ./run-webapp.sh
WHISPER_DEVICE=cpu ./run-webapp.sh
WHISPER_COMPUTE_TYPE=int8 ./run-webapp.sh
TRANSCRIPT_MODEL_ROOTS=/ruta/a/modelos ./run-webapp.sh
```

`WHISPER_COMPUTE_TYPE` queda en `float16` con CUDA y `int8` con CPU, salvo que lo
definas manualmente.

## Ajustes

```bash
TRANSCRIPT_PHRASE_SILENCE_SECONDS=0.55 ./run-webapp.sh
TRANSCRIPT_PARAGRAPH_SILENCE_SECONDS=1.2 ./run-webapp.sh
TRANSCRIPT_SPEECH_RMS_THRESHOLD=0.0025 ./run-webapp.sh
TRANSCRIPT_ADAPTIVE_RMS_MULTIPLIER=3.0 ./run-webapp.sh
WHISPER_BEAM_SIZE=5 ./run-webapp.sh
WHISPER_CONTEXT_WORDS=24 ./run-webapp.sh
```

Para maxima velocidad puedes bajar `WHISPER_BEAM_SIZE=1`; para mejor continuidad
el default usa `WHISPER_CONTEXT_WORDS=24`. Si aparecen repeticiones por audio
dificil, prueba `WHISPER_CONTEXT_WORDS=0`.

## Pruebas rapidas

Chequeo de sintaxis Python:

```bash
python -m compileall backend scripts
```

Chequeo de sintaxis frontend:

```bash
node --check frontend/static/app.js
```

Prueba de endpoints con la app corriendo:

```bash
./run-webapp.sh
curl -s http://127.0.0.1:8099/api/health
curl -s http://127.0.0.1:8099/api/models
```

En otra terminal puedes verificar que el puerto esta escuchando:

```bash
ss -ltnp 'sport = :8099'
```

## Benchmark de precision

Instala dependencias extra del benchmark:

```bash
python -m pip install -r benchmark-requirements.txt
```

Descarga el dataset AMI de dos hablantes y materializalo en `benchmark_data/`:

```bash
python scripts/download_benchmark_dataset.py
```

Corre el benchmark recomendado:

```bash
python scripts/benchmark_asr.py --model large-v3 --limit 10
```

Corrida completa:

```bash
python scripts/benchmark_asr.py --model large-v3 --limit 50 --configs live-3s-context24,live-3s-beam5
```

El dataset queda en `benchmark_data/ami_2speaker_test/` y los resultados en
`benchmark_results/`. Ambas carpetas estan ignoradas por Git.

Resultado local de referencia con `large-v3`, CUDA/float16 y 10 clips:

| Config | WER | Bag F1 | RTF | Latencia chunk |
| --- | ---: | ---: | ---: | ---: |
| live-3s-beam5 | 0.261 | 0.828 | 0.053 | 0.155s |
| live-3s-context24 | 0.240 | 0.845 | 0.050 | 0.146s |
| live-2s-beam5 | 0.273 | 0.815 | 0.068 | 0.134s |
| live-1s-beam5 | 0.388 | 0.719 | 0.102 | 0.103s |

En una corrida completa de 50 clips, `live-3s-context24` bajo WER de `0.294` a
`0.277` frente a `live-3s-beam5`.

## Publicar en GitHub

El repo debe subir codigo, scripts y estructura de carpetas, pero no modelos ni
datasets generados.

```bash
git add .
git commit -m "Initial Transcript Direct app"
gh repo create transcript-direct --private --source=. --remote=origin --push
```

Para publicarlo como repo publico:

```bash
gh repo create transcript-direct --public --source=. --remote=origin --push
```
