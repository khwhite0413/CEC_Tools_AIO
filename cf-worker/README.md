# CEC All-In-One update proxy (Cloudflare Worker)

Fronts the private `khwhite0413/CEC_Tools_AIO` repo's Releases API so the
desktop client can check for updates and download zipballs **without
shipping a GitHub PAT**. The client only knows the Worker's public URL;
the Worker holds the token as an encrypted secret and does the GitHub
auth on its side.

## Deployed instance

```
https://cec-aio-update.khwhite0413.workers.dev
```

Endpoints:

- `GET /latest` — JSON `{ version, tag_name, html_url, body, published_at, download_endpoint }`
- `GET /download/<tag>` — 302 redirect to a short-lived signed codeload URL

Client code:
- `app/update_checker.py` (hits `/latest`)
- `app/updater.py` (hits `/download/<tag>`)

## Deploy / redeploy

The code in this folder (`worker.js`) is the single source of truth. When
you edit it, redeploy either way below.

### Option A — Cloudflare dashboard (no CLI)

1. https://dash.cloudflare.com → **Workers & Pages** → `cec-aio-update`
2. **Edit code** → paste the full contents of `worker.js` (replace
   whatever's there) → **Save and Deploy**
3. If this is the first deploy, also set the secret:
   - **Settings → Variables and Secrets → Add**
   - Type: **Secret**
   - Variable name: `GITHUB_TOKEN`
   - Value: the fine-grained PAT (see next section)
   - **Deploy**

### Option B — Wrangler CLI

```
npm install -g wrangler
cd cf-worker
wrangler deploy worker.js --name cec-aio-update
wrangler secret put GITHUB_TOKEN --name cec-aio-update
# paste the PAT when prompted
```

## GitHub PAT (fine-grained)

Create at: GitHub → Settings → Developer settings → Personal access
tokens → **Fine-grained tokens** → Generate new token.

- **Token name**: `CEC-AIO-updater-worker`
- **Expiration**: 1 year (rotate before expiry)
- **Repository access**: Only select repositories → `khwhite0413/CEC_Tools_AIO`
- **Repository permissions**:
  - **Contents: Read-only** (allows release asset download)
  - Metadata: Read-only (auto)
  - Everything else: No access

A leaked token can only download this one repo's release ZIPs (same
content that gets shipped to real users). Rotate on suspected leak by
generating a new token and updating the `GITHUB_TOKEN` Worker secret
via the same dashboard/CLI flow.

## Verification

```
curl -s https://cec-aio-update.khwhite0413.workers.dev/latest | jq .
```

Expected:

```json
{
  "version": "1.0.8",
  "tag_name": "v1.0.8",
  "html_url": "https://github.com/khwhite0413/CEC_Tools_AIO/releases/tag/v1.0.8",
  "body": "…markdown release notes…",
  "published_at": "2026-…",
  "download_endpoint": "/download/v1.0.8"
}
```

Error diagnostics:

| Response                                         | Cause                          | Fix                                                      |
| ------------------------------------------------ | ------------------------------ | -------------------------------------------------------- |
| `GITHUB_TOKEN secret not configured on Worker`   | secret name mismatch / unset   | Re-add secret in Cloudflare, exact name `GITHUB_TOKEN`   |
| `GitHub API 401`                                 | PAT expired / revoked          | Generate new fine-grained PAT, update Worker secret      |
| `GitHub API 404`                                 | PAT lacks access to this repo  | Regenerate PAT with `khwhite0413/CEC_Tools_AIO` selected |
| `GitHub API 403`                                 | rate limit (very rare)         | Wait a few min; consider caching key                     |
