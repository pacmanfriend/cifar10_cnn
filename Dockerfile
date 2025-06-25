FROM node:22.15.0-alpine AS web-builder
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web ./
RUN npm run build

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS rust-builder
ARG RUST_VERSION=1.83.0
ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        build-essential \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain ${RUST_VERSION}

WORKDIR /app
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY src ./src
RUN cargo build --release

FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04
RUN apt-get update \
    && apt-get install -y --no-install-recommends cuda-nvrtc-12-8 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=rust-builder /app/target/release/cifar10_cnn ./
COPY --from=web-builder /web/build ./web/build

EXPOSE 8080
CMD ["./cifar10_cnn", "--port", "8080"]
