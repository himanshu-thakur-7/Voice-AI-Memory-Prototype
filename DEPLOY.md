# Deploying — Vercel (frontend) + Render (backend)

This repo is split into two deployables:

| Piece | Where | Service type |
|---|---|---|
| Static HTML demo console | **Vercel** | Static |
| FastAPI HTTP API (`/api/personas`, `/api/seed`, `/api/token`, `/api/verify`) | **Render** | Web Service (free tier) |
| LiveKit Agents worker (the voice AI itself) | **Render** | Background Worker (free tier) |

Both Render services are defined in [`render.yaml`](./render.yaml); both share the same Python env. The Vercel build is defined in [`vercel.json`](./vercel.json) and [`frontend/build.sh`](./frontend/build.sh).

Hosted infra you'll plug in (all have free tiers):
- LiveKit Cloud — `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET`
- Neo4j Aura — `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD`
- OpenAI — `OPENAI_API_KEY` (LLM + TTS now)
- Deepgram — `DEEPGRAM_API_KEY` (STT)
- ElevenLabs — `ELEVENLABS_API_KEY` (vestigial; kept for compat in case you swap TTS back)

---

## Step 1 — Push to GitHub

The repo is already at https://github.com/himanshu-thakur-7/Voice-AI-Memory-Prototype. Make sure your latest local commits are pushed before deploying — both Render and Vercel auto-deploy from `main`.

```bash
cd "/Users/little_beast/Desktop/Voice AI Prototype"
git status
git add -A
git commit -m "Deploy: Vercel + Render scaffolding"
git push origin main
```

---

## Step 2 — Deploy the backend to Render

### 2.1 Create the services from the blueprint

1. Sign in at [render.com](https://render.com).
2. New → **Blueprint** → select your GitHub repo `himanshu-thakur-7/Voice-AI-Memory-Prototype`.
3. Render reads `render.yaml` and proposes two services: `cognitive-voice-api` (web) and `cognitive-voice-agent` (worker). Approve.

### 2.2 Fill in the secret env vars

Render will create the services but leave the secret env vars blank (everything marked `sync: false` in `render.yaml`). For **each** of the two services, open Environment and paste:

| Key | Value |
|---|---|
| `OPENAI_API_KEY` | `sk-…` from your OpenAI dashboard |
| `DEEPGRAM_API_KEY` | from your Deepgram dashboard |
| `ELEVENLABS_API_KEY` | from your ElevenLabs dashboard (any valid key — TTS now uses OpenAI but the validator still reads this) |
| `LIVEKIT_URL` | `wss://your-project.livekit.cloud` |
| `LIVEKIT_API_KEY` | `API…` from LiveKit Cloud |
| `LIVEKIT_API_SECRET` | from LiveKit Cloud |
| `NEO4J_URI` | `neo4j+s://xxxxxxxx.databases.neo4j.io` |
| `NEO4J_USER` | usually `neo4j` (or the instance ID) |
| `NEO4J_PASSWORD` | from when you created the Aura instance |

Save. Both services will redeploy automatically.

### 2.3 Verify

The web service exposes a URL like `https://cognitive-voice-api.onrender.com`. Test:

```bash
curl https://cognitive-voice-api.onrender.com/healthz
# → {"ok":true,"ts":…}

curl https://cognitive-voice-api.onrender.com/api/personas
# → {"personas":[{"key":"anil",…},{"key":"first_time",…}]}
```

The worker has no public URL — check the Render logs for `registered worker` to confirm it connected to LiveKit Cloud.

### 2.4 Free-tier caveats

- **Cold start ~30 s.** First request after 15 min idle wakes the dyno. The Vercel frontend shows a small "server up" pill — if it shows "server down" briefly, that's just the cold start.
- **The worker also spins down.** When idle the LiveKit dispatcher will see the worker as offline. The first call attempt warms it back up but adds the same 30 s delay before the agent even tries to speak.
- For a sales/demo run, ping `https://your-api.onrender.com/healthz` a minute before the demo to pre-warm.

---

## Step 3 — Deploy the frontend to Vercel

1. Sign in at [vercel.com](https://vercel.com) → **Add New… → Project**.
2. Import the same GitHub repo.
3. Vercel autodetects `vercel.json`. Framework should be **Other** (we don't use Next/React).
4. Open **Settings → Environment Variables** and add **one** variable:

   | Key | Value |
   |---|---|
   | `BACKEND_URL` | the Render web service URL, e.g. `https://cognitive-voice-api.onrender.com` |

5. Deploy.

The build runs `bash frontend/build.sh`, which substitutes `BACKEND_URL` into `web/static/index.html` and writes the output to `frontend/dist/`. Vercel serves that as the site.

### 3.1 Verify

Open the Vercel URL. The dashboard should load — same UI as local — and the "server up" pill should turn green within a couple of seconds. Pick the **First-time caller** persona, click **Start call**, allow mic access, and you should hear the opener.

---

## Step 4 — Wire LiveKit to allow your Vercel origin

LiveKit Cloud doesn't need any frontend allowlist for connecting to a room — the `/api/token` endpoint mints a JWT and the browser connects directly to `LIVEKIT_URL`. There's nothing to change in LiveKit Cloud itself for this deploy.

If the browser dev tools later show a CORS error pointing at the LiveKit signaling URL, that means the JWT minting hit a network issue, not LiveKit refusing the origin.

---

## Common issues

| Symptom | Likely cause | Fix |
|---|---|---|
| `/healthz` returns 503 | Worker can't reach Neo4j Aura | Check `NEO4J_URI` includes `neo4j+s://`, password is URL-safe |
| First call takes 30 s before agent speaks | Render free-tier cold start | Upgrade to Starter ($7/mo) or pre-warm with `curl /healthz` |
| Agent connects but is silent | OpenAI key missing on **worker** (not just web) | Set `OPENAI_API_KEY` on both Render services — env vars are per-service |
| Verify panel shows "server down" persistently | `BACKEND_URL` mistyped on Vercel | Check Vercel env var, redeploy after fixing |
| All four wow checks stay pending | Aura is empty for this caller | Click **Reset to blank** (first-time) or **Seed graph** (Anil) |

---

## Local-dev parity

The same `web/static/index.html` is used locally and in production. When running locally:

```bash
uvicorn web.server:app --host 127.0.0.1 --port 8000  # serves the static HTML at /
python agent.py dev                                  # registers with LiveKit Cloud
```

The placeholder `__API_BASE_URL__` isn't substituted, so the frontend uses relative paths and everything stays on `localhost:8000`. No code change between local and prod.
