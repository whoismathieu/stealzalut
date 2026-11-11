# Plan d'implémentation v3

Règles communes à toutes les tâches :
- un commit par tâche, message en anglais (`feat: …`, corps qui dit pourquoi), **aucune
  ligne d'attribution** ;
- avant chaque commit, les trois commandes doivent être vertes :
  ```bash
  .venv/Scripts/python.exe -m pytest -q
  .venv/Scripts/python.exe -m ruff check src tests
  .venv/Scripts/python.exe -m mypy src
  ```
- `core/` n'importe jamais Qt ; les textes d'interface sont ceux de `SPEC.md` §4 ;
- mets à jour `STATUS.md` après chaque tâche.

Pièges connus :
- deux tests comptent les méthodes (`tests/test_color_transfer.py` : `len(SUPPORTED_METHODS)
  == 11` ; `tests/test_gui.py` : `window._algo_combo.count() == 11`). N'ajoute pas de
  méthode dans ce plan ;
- `mypy --strict` refuse `cast` superflu et les `np.quantile` non typés : annote avec
  `np.asarray(..., dtype=np.float64)` ;
- polices réglées par feuille de style + `QPushButton.setMenu()` → avertissement
  `QFont::setPointSize` : ouvre les menus à la main comme le fait le menu « Importer » ;
- `EditState` est construit positionnellement à plusieurs endroits de `main_window.py`
  (`EditState(table, params, tau, finishing, separation)`) : n'ajoute des champs qu'**à la fin, avec
  une valeur par défaut**.

---

## T01 — Moteur `look` · **fait** (commit `098396b`)
`src/stealzalut/core/look.py`, `tests/test_look.py` (23 tests).

## T02 — Enregistrement de la méthode · **fait** (commit `098396b`)
`SUPPORTED_METHODS`, `METHOD_LABELS`, `COLOR_MATCHER_METHODS`, `fit_transfer`.

---

## T03 — Curseur « Séparation » dans l'interface · **fait**
`EditState.separation`, curseur `separationSlider` à côté de la liste des méthodes
(visible seulement en méthode `look`), `MainWindow._effective_model` et
`_prepare_model` (remplace `_with_tau`), annulation. Tests : `TestSeparationSlider`
(3 tests) dans `tests/test_gui.py`.

Reste connu : si l'export est lancé alors qu'aucun modèle n'est en cache (cas rare :
l'aperçu précède toujours l'export), `ExportWorker` réajuste le modèle sans appliquer la
séparation. `ExportWorker` n'a pas de paramètre `prepare` ; lui en donner un, sur le
modèle de `TransferWorker`, corrige ce cas.

---

## T04 — Dire les limites · **fait**
`LookDiagnostics`, `look_diagnostics`, `LOW_COLOUR_SHARE`, `WEAK_BAND_TRUST` dans
`core/look.py` ; bandeau dans `MainWindow` (`_look_diagnostics`, `_look_hint_text`,
`_update_mode_hint`). En méthode `look`, le bandeau « Passer en Ambiance » n'apparaît
plus (la méthode protège elle-même les couleurs vives) et son bouton est masqué.
Tests : trois dans `tests/test_look.py` (`test_diagnostics_*`), `TestLookBanner`
(trois) dans `tests/test_gui.py`.

Mesuré sur les photos réelles : cas A et B, part colorée de la source 0,136 → pas de
bandeau ; paire n° 6, 0,011 → bandeau ; paire n° 3, 0,036 → pas de bandeau (juste
au-dessus du seuil de 0,03).

Limite connue : le message « tons absents » ne s'est déclenché sur **aucune** des huit
paires (la courbe de tons envoie toujours les bandes dans la plage de la référence). Il
n'est couvert que par un test qui force `band_trust`.

---

## T05 — Choisir ce que l'on lit dans la référence

**But.** Ignorer ou privilégier des zones de la référence (cadre noir, sujet).

**Fichiers.** `core/look.py` (`zone_weights`, `SPEC.md` §3.2), `gui/history.py`
(`focus`), `gui/main_window.py`, `workers/transfer_worker.py`, tests.

**Étapes.**
1. `zone_weights` + tests (`test_zone_weights_default_is_one`,
   `test_zone_weights_rejects_out_of_range`, `test_labels_match_the_analysis_copy`).
2. Les zones de la référence viennent de `analyze_zones(reference, zone_count)`
   (`core/palette.py`, `analyze_zones(image_rgb, zone_count) -> ZonePalette`, à relire) :
   les calculer une fois par référence et les garder dans la fenêtre.
3. `TransferWorker` : nouveau paramètre `reference_weights: NDArray[np.float64] | None =
   None`, transmis à `fit_look` quand `method == "look"` (dans `_fit`).
4. La clé de cache du modèle (`_method_key`, ≈ l. 1054) doit inclure le `focus` :
   `f"look:{focus}"`, sinon un ancien modèle serait réutilisé.
5. Interface : clic droit sur une zone de la référence dans la vue existante → les trois
   entrées de `SPEC.md` §4 (poids 0, 2, 1).

**Oracle.** `tests/test_look.py::test_reference_weights_change_what_is_read` (existant)
montre l'effet attendu au niveau du moteur.

**Ne pas faire.** Pondérer la *source* (non prototypé).

---

## T06 — Recette `.szl` : mode `look`

**But.** Sauvegarder / recharger / recevoir d'une IA les réglages du mode `look`.

**Fichiers.** `core/recipe.py`, `gui/main_window.py` (`_apply_recipe` ≈ l. 660–730,
calquer `_apply_ambiance_recipe`), `tests/test_recipe.py`.

**Spécification.** `SPEC.md` §3.4. La version reste **2** : le bloc `look` est un ajout
facultatif, les anciennes recettes ne changent pas de sens.

**Tests.** `test_look_recipe_roundtrip`, `test_look_recipe_ignores_links`,
`test_look_separation_out_of_range`, `test_look_focus_unknown_zone`,
`test_v2_recipes_still_parse` (une recette de `docs/audit/data/` doit donner le même
`Recipe` qu'avant, champs nouveaux à leur valeur par défaut).

**Ne pas faire.** Changer `RECIPE_VERSION` ; rendre `look` obligatoire.

---

## T07 — Pack IA du mode `look` : l'IA voit le rendu

**But.** Boucle de correction : rendu → pack → réponse → rendu.

**Fichiers.** `core/ai_pack.py`, `gui/main_window.py` (`_on_prepare_ai_pack` ≈ l. 568),
`tests/test_ai_pack.py`.

**Étapes.**
1. `write_pack` gagne deux paramètres facultatifs en fin de signature :
   `render_rgb: NDArray[np.uint8] | None = None`, `look: LookTransferModel | None = None`.
   S'ils sont donnés : écrire `7_current_render.jpg`, la clé `"look"` de `analysis.json`
   (`SPEC.md` §3.5) et le prompt ci-dessous à la place de `_PROMPT`.
2. La fenêtre passe l'aperçu courant (`self._preview.image_rgb`).

**Prompt (mot pour mot, les `{…}` sont remplis par le code).**
```
# Corriger un rendu StealZaLut (mode « Rendu »)

Tu es coloriste. Trois images comptent :
- `1_source.jpg` : la photo d'origine ;
- `2_reference.jpg` : l'ambiance à atteindre ;
- `7_current_render.jpg` : ce que l'application produit **maintenant**.

L'application a déjà copié la courbe de tons, la dominante de chaque ton et la quantité
de couleur de la référence. Ton rôle : comparer le rendu à la référence **à l'œil** et
dire quoi corriger. Écarts mesurés (rendu → référence) :

{gaps_table}

## Ce que tu peux régler
- `look.separation` (0 à 1.5, actuellement {separation}) : écarte les couleurs proches
  du gris vers celles de la référence. Monte-le si le rendu paraît fade ou uniforme.
- `look.protect_vivid` (0 à 1, actuellement {protect_vivid}) : garde la teinte des
  néons et des couleurs vives de la photo. Monte-le si le rendu ressemble à un filtre.
- `look.focus` : poids des zones de la référence ({reference_ids}, voir
  `4_reference_zones.png`). 0 = ignorer (cadre, sujet hors propos), 2 = privilégier.
- `global.luminance_strength`, `global.chroma_strength`, `global.mix` (0 à 1).
- `finishing` : `saturation`, `vibrance` (−1 à 1), et les roues `shadows`, `midtones`,
  `highlights` (`hue` en degrés, `amount` de 0 à 1, `lightness` de −1 à 1) pour teinter
  une tranche de tons.

## Règles
- Ne propose que ce que tu **vois** en comparant les images.
- Un petit nombre de corrections franches vaut mieux que beaucoup de retouches timides.
- Deux régions de même couleur ne peuvent pas être traitées différemment : dis-le dans
  `description` si c'est ce qui manque.

## Réponse
Un seul bloc JSON :
{example_json}
```
`{example_json}` : une recette valide avec `"format": "stealzalut.recipe"`, `"version":
2`, `"analysis_id"`, `"mode": "look"`, `"name"`, `"description"`, `"look"`, `"global"`,
`"finishing"`.

**Tests.** `test_look_pack_contains_the_render`, `test_look_prompt_lists_reference_ids`,
`test_look_prompt_example_parses` (l'exemple du prompt passe `parse_recipe`).

**Validation obligatoire, hors tests.** Donner un vrai pack à une vraie IA (cas A et B),
importer sa réponse, joindre la planche avant / après à `ACCEPTANCE.md`. **Ce prompt
n'a jamais été soumis à une IA.**

---

## T08 — Méthode par défaut · **fait** (décision de Mathieu du 2026-11-11)
`Settings.color_transfer_method` vaut `"look"` (`src/stealzalut/config.py`, valeur du
champ et valeur de repli de `COLOR_TRANSFER_METHOD`), ainsi que `.env.example`.

Piège : le fichier `.env` local (hors Git) fixe `COLOR_TRANSFER_METHOD` et **prime sur
le code**. Deux tests d'interface supposaient l'ancien défaut ; ils choisissent
maintenant leur méthode explicitement.

---

## Phase 2 — non planifiée
Sorties avec masques (une LUT par région + caches PNG, ou image gradée). Préalable :
une segmentation fiable (pinceau, ou modèle ONNX optionnel avec licence vérifiée) et une
planche qui montre un gain visible sur un cas réel. Ne pas commencer sans cela.
