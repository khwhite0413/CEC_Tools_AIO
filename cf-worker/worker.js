/**
 * CEC All-In-One update proxy (Cloudflare Worker).
 *
 * Deployed at: https://cec-aio-update.khwhite0413.workers.dev
 *
 * Two endpoints:
 *   GET /latest
 *       → JSON { version, tag_name, html_url, body, download_endpoint }
 *   GET /download/<tag>
 *       → 302 redirect to a short-lived signed URL that streams the
 *         release zipball straight from codeload.github.com.
 *
 * The GitHub PAT lives ONLY as the GITHUB_TOKEN secret on this Worker.
 * The client never sees it and never hits the GitHub API directly. PAT
 * is fine-grained, scoped to Contents:Read on khwhite0413/CEC_Tools_AIO
 * only — a leak would expose read-only access to that one repo's
 * releases and nothing else.
 *
 * See cf-worker/README.md for deploy / rotate instructions.
 */

const REPO = "khwhite0413/CEC_Tools_AIO";

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname.replace(/\/+$/, "") || "/";

    const cors = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, OPTIONS",
      "Access-Control-Max-Age": "86400",
    };
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: cors });
    }
    if (request.method !== "GET") {
      return new Response("Method Not Allowed", { status: 405, headers: cors });
    }

    const token = env.GITHUB_TOKEN;
    if (!token) {
      return json({ error: "GITHUB_TOKEN secret not configured on Worker" }, 500, cors);
    }
    const gh = {
      "Authorization": `Bearer ${token}`,
      "Accept": "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "cec-aio-update-worker",
    };

    // ---- /latest ---------------------------------------------------
    if (path === "/latest") {
      const r = await fetch(
        `https://api.github.com/repos/${REPO}/releases/latest`,
        { headers: gh },
      );
      if (!r.ok) {
        return json({ error: `GitHub API ${r.status}`, body: await r.text() },
                    r.status === 404 ? 404 : 502, cors);
      }
      const rel = await r.json();
      const tag = rel.tag_name || "";
      const version = tag.replace(/^v/, "");
      return json({
        version,
        tag_name: tag,
        html_url: rel.html_url || "",
        body: rel.body || "",
        published_at: rel.published_at || "",
        download_endpoint: `/download/${encodeURIComponent(tag)}`,
      }, 200, {
        ...cors,
        // 60 s edge cache — enough to smooth request spikes, short
        // enough that a fresh release is visible almost immediately.
        "Cache-Control": "public, max-age=60",
      });
    }

    // ---- /download/<tag> -------------------------------------------
    const m = path.match(/^\/download\/(.+)$/);
    if (m) {
      const tag = decodeURIComponent(m[1]);
      // GitHub's zipball endpoint 302s to a signed codeload URL that
      // needs no auth for a few minutes. Return that Location straight
      // to the client so the actual ZIP transfer bypasses this Worker
      // entirely (saves Worker CPU + doesn't count against Worker
      // bandwidth, which matters if we ever hit paid tiers).
      const r = await fetch(
        `https://api.github.com/repos/${REPO}/zipball/${encodeURIComponent(tag)}`,
        { headers: gh, redirect: "manual" },
      );
      if (r.status === 301 || r.status === 302 || r.status === 307) {
        const location = r.headers.get("Location");
        if (!location) {
          return new Response("GitHub redirect missing Location header", { status: 502, headers: cors });
        }
        return Response.redirect(location, 302);
      }
      if (r.ok) {
        // Fallback: stream body through if GitHub didn't redirect.
        return new Response(r.body, {
          status: 200,
          headers: {
            ...cors,
            "Content-Type": "application/zip",
            "Content-Disposition": `attachment; filename="CEC_Tools_AIO-${tag}.zip"`,
          },
        });
      }
      return new Response(`GitHub ${r.status} on zipball`, { status: 502, headers: cors });
    }

    // ---- root info page --------------------------------------------
    if (path === "/") {
      return new Response(
        `CEC All-In-One update proxy\nRepo: ${REPO}\nEndpoints:\n  GET /latest\n  GET /download/<tag>\n`,
        { status: 200, headers: { ...cors, "Content-Type": "text/plain; charset=utf-8" } },
      );
    }

    return new Response("Not Found", { status: 404, headers: cors });
  },
};

function json(obj, status, extraHeaders) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: {
      ...(extraHeaders || {}),
      "Content-Type": "application/json; charset=utf-8",
    },
  });
}
