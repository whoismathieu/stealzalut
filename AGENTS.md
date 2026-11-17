# StealZaLut — Entry Point for a New Session

Read this file first, and only this file. It tells you what the project is, the rules, the commands, and **which part of which document to open for the task at hand**. Do not read `ARCHITECTURE.md` (35 kB) or the source tree from end to end: open the section listed in §4.

## 1. What this is

A Windows desktop application (Python 3.10, PySide6). The user drops a **source** photo and a **reference** photo; the application computes the colour rendering of the reference, previews it on the source, and exports a **3D LUT (`.cube`)** to be used in Lightroom Classic / Camera Raw (as a profile) or DaVinci Resolve. The user is a French-speaking photographer and developer who works mostly on iPhone ProRAW (`.DNG`) night scenes.

## 2. Rules (non-negotiable)

1. **Language.** Talk to the user in **French**. Interface texts and user-facing messages are in French. Code, identifiers, docstrings, commit messages and every `.md` file are in **English**.
2. **No AI attribution**, anywhere: no `Co-Authored-By`, no "Generated with…", no model or assistant name in commits, comments or files.
3. **Git.**
   - Commit directly on `main`; no feature branches, no history rewriting, no `--amend` on pushed commits.
   - **Push only when the user explicitly says so**, each time. Run `git push origin main` and never print its remote line unmasked.
   - **Never** run `git remote -v` or `git config --list`, and never read `.git/config`: the remote URL holds a credential.
   - Commit messages: `feat:` / `fix:` / `docs:` title, then a body that explains **why**.
   - Never commit the user's photos, `.cube` files, or the untracked `.claude/`, `project2txt.sh`, `stealzalut_code.txt`.
4. **Before every commit**, all three must pass with no error and no warning:

   ```bash
   .venv/Scripts/python.exe -m pytest -q
   .venv/Scripts/python.exe -m ruff check src tests
   .venv/Scripts/python.exe -m mypy src
   ```

5. **Documents.** `AGENTS.md` (this file), `README.md` (users) and `ARCHITECTURE.md` (developers) are the only `.md` files. Never create another one. A change of behaviour updates `README.md` and `ARCHITECTURE.md` in the same commit; update §6 of this file when the state of the project changes.
6. **Disk.** The system drive is over 95 % full. Captures, `.npy` files and any measurement output go to a temporary folder **outside** the repository, deleted right after use. Never install a package without asking (there is no scipy; do not add it).
7. **Network.** The application is offline. The single exception is `gui/incoming_image.download` (an image address the user dropped or pasted). Add no other network access.
8. **Honesty of measurements.** Never state that a rendering matches Lightroom without a measurement; say what was measured, on which files, and what was not. Never pick test images at random: ask the user which source / reference pair to use.

## 3. Architecture in ten lines

```text
src/stealzalut/
├── core/      colour science, engines, LUT baking, recipes. MUST NOT import PySide6.
├── workers/   QThread wrappers around core (preview, export).
├── gui/       PySide6 widgets, undo history, themes, preferences.
├── utils/     image reading (WideImage, RAW/DNG development, Adobe .dcp profiles), paths.
└── devtools/  smoke_gui (drives the real window), ground_truth (bench).
tests/         one file per module, 798 tests, about 70 s.
```

Pipeline: file → `utils/image_io.load_image_rgb` → `WideImage` (sRGB array + unclipped Oklab analysis copy + native wide-gamut pixels) → engine in unclipped **Oklab** → `core/grading.build_grid` (the single baking path) → `.cube` in sRGB, Display P3, Adobe RGB or Rec.709.

## 4. Where to look, by task

Open only the listed section or file.

| Task touches | Read | Code |
| :--- | :--- | :--- |
| RAW / DNG development, match with Lightroom | `ARCHITECTURE.md` §3.1 | `utils/image_io.py`, `utils/camera_profile.py` |
| Colour spaces, ICC, `.cube` export space | `ARCHITECTURE.md` §3 | `core/rgb_spaces.py`, `core/grading.py`, `core/lut_apply.py` |
| The « Rendu » engine (`look`) | `ARCHITECTURE.md` §4.1 | `core/look.py` |
| The « Zones » engine | `ARCHITECTURE.md` §4.2 | `core/palette.py`, `core/matching.py`, `core/zone_transfer.py` |
| Recipes `.szl`, packs for an external AI | `ARCHITECTURE.md` §5 | `core/recipe.py`, `core/ai_pack.py`, `core/look_pack.py` |
| Bench, measured findings, oracles | `ARCHITECTURE.md` §6 | `devtools/ground_truth.py`, `tests/test_look.py` |
| Any change to the window | `ARCHITECTURE.md` §7 (pitfalls 3, 7 to 12) | `gui/main_window.py`, `gui/widgets/` |
| Themes, colours of the interface | `ARCHITECTURE.md` §7 pitfall 9 | `gui/theming.py`, `gui/styles/theme.qss`, `gui/styles/themes/*.json` |
| Drop / paste of images | `ARCHITECTURE.md` §7 pitfall 12 | `gui/incoming_image.py`, `gui/widgets/image_drop_zone.py` |
| What the user sees, workflows with Lightroom | `README.md` §2, §3 | |

## 5. Invariants that break things when forgotten

- `len(SUPPORTED_METHODS) == 11` and the method combo has 11 entries (tests check both).
- `EditState` (`gui/history.py`) is built positionally: a new field goes **last, with a default**.
- Never `QPushButton.setMenu()`: open each `QMenu` by hand (`menu.exec(...)`), otherwise Qt prints `QFont::setPointSize` warnings. Font sizes are in `px` in the stylesheet.
- Colours of the interface live in `gui/styles/themes/*.json`, never in `theme.qss` (a template with `@tokens`) nor in the code. A token added to the template is added to `graphite.json` in the same change.
- `PROPHOTO_RGB` is never offered as a `.cube` space. `build_grid` is the only baking path, for every space.
- `rawpy.postprocess` always gets `adjust_maximum_thr=0.0`. Without it every photo is brightened by a factor of its own.
- Image bands are developed on threads: use `image_io._transform`, not `@`, for matrix products there (the linear-algebra library stalls).
- `WideImage` loses `analysis_lab` on slicing or resizing: pass the object itself to the engine.
- A `MainWindow` built without `preferences` keeps choices in memory: tests never touch the user's settings.

## 6. State of the project (2026-11-17, commit `ae34faa`)

Done and pushed, most recent first:

1. Images dragged from a browser or pasted (Ctrl+V, right-click menu).
2. DNG developed with Adobe's own « Adobe Standard » `.dcp` profile found on the machine; white level no longer rescaled. Measured on two iPhone 15 Pro Max ProRAW files against Lightroom Classic: RMS ΔE Oklab 0.016, saturation within 4 % on the file that has a plain export. The user confirmed the `.cube` result in Lightroom.
3. The banner of a RAW source lists the values to enter in Camera Raw's « Créer un profil » window.
4. Themes as JSON palettes (Graphite default, Noir, Darkroom), « Apparence » menu, six Lightroom greys behind the images, French labels.
5. Toolbars that wrap, zoom and pan on images, foldable sections.

Known limits, not started:

- Only one camera measured. RAW files that are not DNG, and cameras without an installed Adobe profile, use the fallback development (`rendering = "neutral-raw"`), less exact.
- A 12 MP DNG takes about 6.5 s to load with the profile (3 s without).
- No light theme: `CorrespondenceView`, the colour wheels and the tone curve paint with hard-coded dark colours.
- AVIF images are not read. A drag from a real browser was tested by simulation and by a real download, not by hand.
- The Gamut value advised for Camera Raw (« Écrêtage ») follows from how the `.cube` is built; it was not compared with the other value.

## 7. How to work here efficiently

- **Search, then read a range.** Find the symbol with a search, then read the 30 to 60 lines around it. `gui/main_window.py` is over 2 000 lines: never read it whole.
- **Check the window for real.** After a change to the interface, run the smoke tool into a temporary folder, look at the capture, check that no `QFont::setPointSize` line is printed, then delete the folder:

  ```bash
  PYTHONPATH=src .venv/Scripts/python.exe -m stealzalut.devtools.smoke_gui SOURCE REFERENCE OUT_FOLDER [look|zones]
  ```

- **Do not run `ruff format` on the whole tree**: it rewrites files that are not yours. Format only the files you created.
- **Shell.** Git Bash on Windows. A heredoc mangles backslashes and apostrophes: write helper scripts to a file outside the repository and run them. Deleting with `rm -rf "$VAR"` is refused by the harness; use `"${VAR:?}"`.
- **Lightroom is evidence.** A TIFF exported by Lightroom embeds its develop settings as XMP (`crs:CameraProfile`, sliders): read them instead of asking. Adobe's camera profiles are in `C:\ProgramData\Adobe\CameraRaw\CameraProfiles\Adobe Standard`.
- **End of a task.** Run the three checks, update the two documents and §6 above, commit, then ask whether to push.
