// The violet worker: the API, the media proxy, and the static app.
// See web/README.md for the contract.

import { routeApi } from "./api.js";
import { identify } from "./auth.js";
import { errorResponse } from "./http.js";
import { serveMedia } from "./media.js";

export async function handle(request, env, deps = {}) {
  const url = new URL(request.url);
  try {
    if (url.pathname.startsWith("/api/")) {
      const email = await identify(request, env, deps);
      return await routeApi(request, env, url, email);
    }
    if (url.pathname.startsWith("/media/")) {
      await identify(request, env, deps);
      return await serveMedia(request, env, url);
    }
    return await env.ASSETS.fetch(request);
  } catch (err) {
    return errorResponse(err);
  }
}

export default {
  fetch(request, env) {
    return handle(request, env);
  },
};
