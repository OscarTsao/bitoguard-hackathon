FROM node:22-bookworm-slim AS deps

WORKDIR /app

COPY bitoguard_frontend/package.json bitoguard_frontend/package-lock.json ./
RUN npm ci --omit=dev


FROM node:22-bookworm-slim AS builder

WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY bitoguard_frontend/ ./

ARG BITOGUARD_INTERNAL_API_BASE=http://backend:8001
ENV BITOGUARD_INTERNAL_API_BASE=${BITOGUARD_INTERNAL_API_BASE} \
    NEXT_TELEMETRY_DISABLED=1

RUN mkdir -p public && npm run build


FROM node:22-bookworm-slim AS runner

WORKDIR /app

ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME=0.0.0.0 \
    BITOGUARD_INTERNAL_API_BASE=http://backend:8001 \
    NEXT_TELEMETRY_DISABLED=1

# standalone server + static assets — node_modules not needed
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

EXPOSE 3000

CMD ["node", "server.js"]
