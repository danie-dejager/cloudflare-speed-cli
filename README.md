# cloudflare-speed-cli

[![Rust](https://img.shields.io/badge/rust-1.81+-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

A CLI tool that displays network speed test results from Cloudflare's speed test service in a TUI interface.

## Features

- **Interactive TUI**: Real-time charts and statistics
- **Speed Tests**: Measures download/upload throughput, idle latency, and loaded latency
- **History**: View and manage past test results
- **Export**: Save results as JSON
- **Text/JSON Modes**: Headless operation for scripting

## Installation

### Linux (All Distributions)

Download the static binary (works on any Linux distribution):

```bash
# For x86_64 systems
curl -L https://github.com/kavehtehrani/cloudflare-speed-cli/releases/latest/download/cloudflare-speed-cli_-x86_64-unknown-linux-musl.tar.xz | tar xJ
sudo mv cloudflare-speed-cli /usr/local/bin/

# For ARM64 systems (Raspberry Pi, etc.)
curl -L https://github.com/kavehtehrani/cloudflare-speed-cli/releases/latest/download/cloudflare-speed-cli_-aarch64-unknown-linux-musl.tar.xz | tar xJ
sudo mv cloudflare-speed-cli /usr/local/bin/
```

### macOS

```bash
# For Intel Macs
curl -L https://github.com/kavehtehrani/cloudflare-speed-cli/releases/latest/download/cloudflare-speed-cli_-x86_64-apple-darwin.tar.xz | tar xJ
sudo mv cloudflare-speed-cli /usr/local/bin/

# For Apple Silicon (M1/M2/M3)
curl -L https://github.com/kavehtehrani/cloudflare-speed-cli/releases/latest/download/cloudflare-speed-cli_-aarch64-apple-darwin.tar.xz | tar xJ
sudo mv cloudflare-speed-cli /usr/local/bin/
```

### Windows

1. Download `cloudflare-speed-cli_-x86_64-pc-windows-msvc.zip` from [GitHub Releases](https://github.com/kavehtehrani/cloudflare-speed-cli/releases)
2. Extract the ZIP file
3. Move `cloudflare-speed-cli.exe` to a directory in your PATH (e.g., `C:\Windows\System32` or add a custom directory to PATH)

### From Source (Cargo)

```bash
cargo install --git https://github.com/kavehtehrani/cloudflare-speed-cli --features tui
```


## Usage

Run with the TUI (default):

```bash
cloudflare-speed-cli
```

Text output mode:

```bash
cloudflare-speed-cli --text
```

JSON output mode:

```bash
cloudflare-speed-cli --json
```

## TUI Controls

### Dashboard Tab
- `q` / `Ctrl-C`: Quit
- `r`: Rerun test
- `p`: Pause/Resume
- `s`: Save JSON to auto-save location
- `a`: Toggle auto-save
- `tab`: Switch tabs (Dashboard, History, Help)
- `?`: Show help

### History Tab
- `↑/↓` or `j/k`: Navigate through historical runs
- `e`: Export selected run as JSON (to current directory)
- `c`: Export selected run as CSV (to current directory)
- `y`: Copy last exported file path to clipboard
- `d`: Delete selected history item
- `tab`: Switch tabs

### Export Options

- **`s` (Save JSON)**: On dashboard, saves the current/last run to the default auto-save location
- **`e` (Export JSON)**: In history tab, exports the selected historical run as JSON with full absolute path shown
- **`c` (Export CSV)**: In history tab, exports the selected historical run as CSV with full absolute path shown
- **`y` (Yank/Copy)**: In history tab, copies the last exported file's absolute path to clipboard

The exported files are saved to the current working directory with filenames based on the test timestamp and measurement ID. The full absolute path is displayed and can be copied to clipboard.

## Source

Uses endpoints from https://speed.cloudflare.com/
