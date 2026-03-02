ARG RUST_VERSION=1.88
ARG ALPINE_VERSION=3.21

# ---------------------------------------------------------------------------
# Stage 1 - Build a static binary with musl
# ---------------------------------------------------------------------------
FROM rust:${RUST_VERSION}-alpine${ALPINE_VERSION} AS builder

RUN apk add --no-cache musl-dev

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src/ src/

RUN cargo build --release && \
    cp target/release/cloudflare-speed-cli /cloudflare-speed-cli

# ---------------------------------------------------------------------------
# Stage 2 - Minimal runtime image
# ---------------------------------------------------------------------------
FROM alpine:${ALPINE_VERSION}

RUN apk add --no-cache ca-certificates

COPY --from=builder /cloudflare-speed-cli /usr/local/bin/cloudflare-speed-cli

ENV TERM=xterm-256color

ENTRYPOINT ["cloudflare-speed-cli"]
