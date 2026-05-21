# prose-decorate

Markdown on stdin -> Fish Audio S2-Pro prosody-tagged plain text on stdout.

Closes the loop between [`substack-url-tool`](https://github.com/jonathanmoregard/substack-url-tool)
and [`tts-tool`](https://github.com/jonathanmoregard/tts-tool):

```sh
substack-url-tool --format markdown "$URL" | prose-decorate | tts-tool -o out.mp3
```

The middle stage is an LLM that reads the article's Markdown and inserts
Fish-S2-Pro `[bracket]` tags — `[short pause]`, `[long pause]`,
`[emphasis]`, `[thoughtfully]`, etc. — at points where the prose's
own cues (headings, em-dashes, blockquotes, **bold**) say a human
narrator would change cadence. Without this stage, S2-Pro reads the
article as a flat monologue.

## How it works

```
markdown (stdin)
  ├─ deterministic strip:
  │    code blocks   -> [code omitted]
  │    inline code   -> [code]
  │    images        -> alt text only
  │    tables        -> [table omitted]
  │    footnotes     -> dropped
  │    links         -> link text only
  ├─ paragraph chunker (target <= 4000 chars; sentence-boundary fallback
  │    when a paragraph alone exceeds the limit)
  ├─ per-chunk Claude Sonnet 4.6 call (temperature=0)
  │    previous chunk's last paragraph passed in <ctx-prev-NONCE>...</>
  │    for tonal continuity without cross-chunk state on output
  ├─ guards:
  │    strip_tags(output) MUST equal normalize_whitespace(input)
  │    tag bytes MUST be <= 0.5 * len(input)
  │    3 retries; persistent failure -> passthrough that chunk
  └─ tagged text (stdout)
```

## Exit codes

| Code | Meaning |
|---|---|
| 0 | All chunks decorated cleanly |
| 10 | Partial passthrough (>=1 chunk fell back to undecorated) |
| 20 | Full passthrough (LLM unavailable, no chunks decorated) |
| 1 | argv / invalid input |
| 2 | Empty input |
| 3 | Refusing to write binary to TTY (n/a; text output) — reserved |

Use `set -e` in shell or check exit code if you need to fail the pipeline
on degraded runs.

## Install (dev)

```sh
nix develop
uv sync --all-extras
uv run pytest
uv run prose-decorate --help
```

## Install (Nix flake)

```sh
nix run github:jonathanmoregard/prose-decorate -- -i article.md -o decorated.txt
```

On dellan, the tool is wired into `environment.systemPackages` via the
`listen-tools` module and reads its key from
`config.age.secrets.anthropic-api-key.path` (same secret used by
research-agent + claude-cl-sync).

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | (required) | Anthropic API key |
| `ANTHROPIC_API_KEY_FILE` | — | Path to file containing the key (overrides above) |
| `PROSE_DECORATE_MODEL` | `claude-sonnet-4-6` | Anthropic model id |
| `PROSE_DECORATE_CACHE_DIR` | platformdirs cache | Per-chunk decorated text cache |

## Known limits (v0.1)

- Acronyms (`API`, `LLM`) are read as words ("Ay-Pee-Eye") not initialisms
  by S2-Pro. The decorator does not expand them.
- Identifiers (`kubectl`, `numpy`) may be mispronounced. Strip or expand
  in the source if it matters.
- Inline `` `code` `` is replaced with `[code]` (TTS would otherwise
  read punctuation literally).
- Cache cascades on edits: editing chunk N forces chunks N and N+1 to
  re-decorate (the rolling-context window is 1 paragraph deep). Acceptable
  trade — Claude calls are cheap.
- No streaming output; the whole article must be decorated before stdout
  drains. (Same shape as `tts-tool`.)

## Why this exists

Fish Audio's S2-Pro supports `[bracket]` natural-language prosody tags
that meaningfully change delivery (pauses, emphasis, emotion). Raw
article text doesn't have those cues — they live in the Markdown
formatting that the TTS-side never sees. This tool is the LLM-judgment
layer that translates between the two.
