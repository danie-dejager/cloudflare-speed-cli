#!/usr/bin/env sh
set -eu

REPO="kavehtehrani/cloudflare-speed-cli"
BIN="cloudflare-speed-cli"

VERSION="${VERSION:-$(curl -fsSL \
  https://api.github.com/repos/${REPO}/releases/latest \
  | sed -n 's/.*"tag_name": "\(.*\)".*/\1/p')}"

[ -n "$VERSION" ] || { echo "Could not resolve latest version" >&2; exit 1; }

case "$(uname -s)" in
  Linux) OS="unknown-linux-musl" ;;
  Darwin) OS="apple-darwin" ;;
  *) echo "Unsupported OS" >&2; exit 1 ;;
esac

case "$(uname -m)" in
  x86_64) ARCH="x86_64" ;;
  arm64|aarch64) ARCH="aarch64" ;;
  *) echo "Unsupported architecture" >&2; exit 1 ;;
esac

FILE="${BIN}_${ARCH}-${OS}.tar.xz"
BASE_URL="https://github.com/${REPO}/releases/download/${VERSION}"

TMP="$(mktemp -d)"
cd "$TMP"

curl -fsSLO "${BASE_URL}/${FILE}"
curl -fsSLO "${BASE_URL}/${FILE}.sha256"

sha256sum -c "${FILE}.sha256"

tar -xJf "$FILE"

INSTALL_DIR="${HOME}/.local/bin"
mkdir -p "$INSTALL_DIR"
install -m 0755 "$BIN" "$INSTALL_DIR/$BIN"

echo "Installed ${BIN} ${VERSION} to ${INSTALL_DIR}"
