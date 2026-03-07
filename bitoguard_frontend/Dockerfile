FROM node:22-bookworm-slim AS deps

WORKDIR /app

COPY bitoguard_frontend/package.json bitoguard_frontend/package-lock.json ./
RUN npm ci

FROM node:22-bookworm-slim AS builder

WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY bitoguard_frontend/ ./

ARG BITOGUARD_INTERNAL_API_BASE=http://backend:8001
ENV BITOGUARD_INTERNAL_API_BASE=${BITOGUARD_INTERNAL_API_BASE}

RUN mkdir -p public \
    && npm run build

FROM node:22-bookworm-slim AS runner

WORKDIR /app

ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME=0.0.0.0 \
    BITOGUARD_INTERNAL_API_BASE=http://backend:8001

COPY --from=builder /app/package.json /app/package-lock.json ./
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/next.config.ts ./next.config.ts

EXPOSE 3000

CMD ["npm", "run", "start", "--", "--hostname", "0.0.0.0", "--port", "3000"]

