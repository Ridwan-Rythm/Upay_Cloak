# Fonts (self-hosted)

The UI no longer loads Google Fonts. To get the original look offline, download these (both SIL Open Font License 1.1,
so they may be bundled in the repo) and save them in this folder with exactly these names:

| File | Family | Source |
|---|---|---|
| `PlusJakartaSans-var.woff2` | Plus Jakarta Sans (variable, wght 400-800) | https://github.com/tokotype/PlusJakartaSans |
| `NotoSansBengali-var.woff2` | Noto Sans Bengali (variable, wght 400-700) | https://github.com/notofonts/bengali |

Until they are added the browser uses an installed copy if present, otherwise `system-ui` (Bangla then depends on the
device having a Bengali-capable font). Status: **files NOT bundled yet** (the audit sandbox had no network); see docs/HUMAN_TODO.md.
