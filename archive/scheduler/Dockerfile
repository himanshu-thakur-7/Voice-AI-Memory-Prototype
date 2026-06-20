# ── build ────────────────────────────────────────────────────────────────────
FROM golang:1.22-alpine AS build
WORKDIR /src
# Cache deps first for fast rebuilds.
COPY go.mod go.sum* ./
RUN go mod download 2>/dev/null || true
COPY . .
RUN CGO_ENABLED=0 go build -o /out/scheduler ./cmd/scheduler

# ── runtime (distroless-ish minimal) ─────────────────────────────────────────
FROM alpine:3.20
RUN adduser -D -u 10001 app
COPY --from=build /out/scheduler /usr/local/bin/scheduler
USER app
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/scheduler"]
