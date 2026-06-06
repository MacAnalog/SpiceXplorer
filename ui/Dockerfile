# syntax=docker/dockerfile:1.7
# =============================================================================
# SpiceXplorer frontend — Next.js 15 (standalone output).
#
# Three stages (deps / builder / runner). The runner ships only the
# self-contained .next/standalone server + static assets — no dev toolchain,
# no full node_modules — so the image stays small. Runs as a non-root user.
#
# Build context = ui/.   Built via:  docker compose build frontend
# The browser hits :4000; Next proxies /api -> BACKEND_URL (the backend service)
# via next.config.mjs rewrites. BACKEND_URL is baked at build time (compose sets
# it to the internal service URL); override with the build arg to retarget.
# =============================================================================
ARG NODE_TAG=22-slim

# ---- deps: install node_modules from the lockfile ----------------------------
FROM node:${NODE_TAG} AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefer-offline

# ---- builder: produce the standalone build -----------------------------------
FROM node:${NODE_TAG} AS builder
WORKDIR /app
ARG BACKEND_URL=http://backend:8000
ENV BACKEND_URL=${BACKEND_URL} \
    NEXT_TELEMETRY_DISABLED=1
COPY --from=deps /app/node_modules ./node_modules

# Copy the whole build context (.dockerignore excludes node_modules/, .next/,
# env files). A selective allowlist is fragile — forgetting a config file (e.g.
# tailwind.config.ts) makes the build succeed but ship unstyled output.
COPY . .

# Build with Next.js cache mount — .next/cache persists compiled modules and the
# webpack cache across builds, speeding up rebuilds when only source changes.
# Note: this cache can pin stale output across a config change; if styling/build
# looks wrong after editing a config, prune it:
#   docker builder prune --filter type=exec.cachemount
RUN --mount=type=cache,target=/app/.next/cache \
    npm run build

# ---- runner: minimal production image ----------------------------------------
FROM node:${NODE_TAG} AS runner
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=4000 \
    HOSTNAME=0.0.0.0
RUN groupadd -g 1001 nodejs && useradd -u 1001 -g nodejs -m nextjs
# .next/standalone bundles a minimal server.js + only the needed node_modules;
# public/ and .next/static must be copied alongside it (standalone omits them).
COPY --from=builder --chown=nextjs:nodejs /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
USER nextjs
EXPOSE 4000
CMD ["node", "server.js"]
