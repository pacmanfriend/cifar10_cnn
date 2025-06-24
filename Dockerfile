FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ARG RUST_VERSION=1.83.0
ARG NODE_VERSION=22.15.0

ENV CARGO_HOME=/usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
ENV PATH=/usr/local/cargo/bin:/usr/local/node/bin:${PATH}

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        build-essential \
        pkg-config \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain ${RUST_VERSION} \
    && chmod -R a+w ${CARGO_HOME} ${RUSTUP_HOME}

RUN arch="$(uname -m)" \
    && case "${arch}" in \
        x86_64) node_arch="x64" ;; \
        aarch64) node_arch="arm64" ;; \
        *) echo "unsupported architecture: ${arch}" >&2; exit 1 ;; \
    esac \
    && curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/SHASUMS256.txt" -o /tmp/node-shasums.txt \
    && node_tar="$(awk -v node_arch="${node_arch}" '$2 ~ "node-v.*-linux-" node_arch ".tar.xz$" { print $2; exit }' /tmp/node-shasums.txt)" \
    && curl -fsSLO "https://nodejs.org/dist/v${NODE_VERSION}/${node_tar}" \
    && grep " ${node_tar}$" /tmp/node-shasums.txt | sha256sum -c - \
    && mkdir -p /usr/local/node \
    && tar -xJf "${node_tar}" -C /usr/local/node --strip-components=1 \
    && rm -f "${node_tar}" /tmp/node-shasums.txt

COPY web/package.json web/package-lock.json ./web/
RUN cd web && npm ci

COPY web ./web
RUN cd web && npm run build

COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY src ./src

RUN cargo build --release

EXPOSE 8080

CMD ["./target/release/cifar10_cnn", "--port", "8080"]
