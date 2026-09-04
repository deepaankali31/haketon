# Deploy online in under 2 minutes

`web/index.html` is fully self-contained: no build step, no server, no
external CDN, no fetch() calls (the precomputed data is embedded directly
in the file). Any of these will work as-is:

## Option A — Netlify (drag and drop, fastest)
1. Go to https://app.netlify.com/drop
2. Drag the `web/` folder onto the page.
3. Netlify gives you a public URL immediately. Done.

## Option B — GitHub Pages
1. Push this repo to GitHub.
2. In repo Settings → Pages, set source to the `web/` folder on your main
   branch (or copy `web/index.html` to the repo root as `index.html`).
3. Your site is live at `https://<username>.github.io/<repo>/`.

## Option C — Vercel
1. `npm i -g vercel` (or use the Vercel web UI → "Add New Project").
2. From the repo root: `vercel --prod` and point the output directory at
   `web/`.

No environment variables, API keys, or backend are required — the entire
interactive artifact runs client-side.
