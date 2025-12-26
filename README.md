# cloudflare-speed-cli

[![Rust](https://img.shields.io/badge/rust-1.81+-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

A CLI tool that displays network speed test results from Cloudflare's speed test service in a TUI interface.

## Features

- **Interactive TUI**: Real-time charts and statistics similar to `btop`
- **Speed Tests**: Measures download/upload throughput, idle latency, and loaded latency
- **History**: View and manage past test results
- **Export**: Save results as JSON
- **Text/JSON Modes**: Headless operation for scripting

## Usage

Run with the TUI (default):
```bash
cargo run --features tui
```

Text output mode:
```bash
cargo run -- --text
```

JSON output mode:
```bash
cargo run -- --json
```

## TUI Controls

- `q` / `Ctrl-C`: Quit
- `r`: Rerun test
- `p`: Pause/Resume
- `s`: Save JSON manually
- `a`: Toggle auto-save
- `tab`: Switch tabs (Dashboard, History, Help)
- `↑/↓` or `j/k`: Navigate history
- `d`: Delete selected history item
- `?`: Show help

## Source

Uses endpoints from https://speed.cloudflare.com/

