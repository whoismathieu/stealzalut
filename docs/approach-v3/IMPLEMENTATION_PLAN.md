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
  (`EditState(table, params, tau, finishing)`) : n'ajoute des champs qu'**à la fin, avec
  une valeur par défaut**.

---

## T01 — Moteur `look` · **fait** (commit `098396b`)
`src/stealzalut/core/look.py`, `tests/test_look.py` (23 tests).

## T02 — Enregistrement de la méthode · **fait** (commit `098396b`)
`SUPPORTED_METHODS`, `METHOD_LABELS`, `COLOR_MATCHER_METHODS`, `fit_transfer`.

---

## T03 — Curseur « Séparation » dans l'interface

**But.** Régler `LookTransferModel.separation` sans réajuster le modèle, avec annulation.

**Fichiers.** `gui/history.py`, `gui/main_window.py`, `tests/test_gui.py`.

**À lire d'abord.** `main_window.py` : `_tau_slider` (≈ l. 326–346), `_on_tau_changed`
(≈ l. 909), `_restore_state` (≈ l. 925–940), `_effective_model` (≈ l. 1031),
`_run_transfer_preview` (≈ l. 1068), l'export (≈ l. 1186–1210), `_with_tau` (fin de
fichier), et la ligne `self._corr_group.setVisible(self._current_method == "zones")`.

**Étapes.**
1. `EditState` : ajouter `separation: float = 1.0` après `finishing`.
2. Créer, à côté du curseur « Transitions », un `QSlider` horizontal `objectName
   "separationSlider"`, plage 0–150, valeur 100, largeur fixe 160, précédé du libellé
   `Séparation :` et suivi d'un `QLabel` de valeur (`f"{v:.2f}"`). Infobulle : `SPEC.md` §4.
3. Les trois widgets ne sont visibles que si `self._current_method == "look"` (même
   endroit que la visibilité de `_corr_group`).
4. `valueChanged` → `self._history.push(replace(self._history.current,
   separation=valeur / 100), "separation")` puis relance de l'aperçu, exactement comme
   `_on_tau_changed`.
5. `_effective_model` : si `isinstance(model, LookTransferModel)`, retourner
   `model.with_separation(state.separation)`.
6. Remplacer `_with_tau(tau, model)` par `_prepare_model(tau, separation, model)` qui
   traite les deux types ; l'utiliser dans `_run_transfer_preview` **et** dans l'export.
7. `_restore_state` : remettre le curseur (signaux bloqués) à `state.separation`.

**Tests à écrire** (`tests/test_gui.py`, même style que les tests du curseur tau) :
- `test_separation_slider_only_visible_for_look`
- `test_separation_slider_changes_the_model` (après changement, `_effective_model` rend
  un modèle dont `separation == 0.5`)
- `test_separation_is_undoable`

**Ne pas faire.** Réajuster le modèle quand le curseur bouge ; toucher à `GradingParams`.

**Terminé quand.** Les trois tests passent ; `prototypes/smoke_look_gui.py` s'exécute
sans ligne `QFont::setPointSize`.

---

## T04 — Dire les limites

**But.** Afficher les deux limites de `DESIGN.md` §8.

**Fichiers.** `core/look.py` (`LookDiagnostics`, `look_diagnostics`, `LOW_COLOUR_SHARE` :
`SPEC.md` §3.1), `gui/main_window.py`, `tests/test_look.py`, `tests/test_gui.py`.

**Algorithme.** `source_colourful = colour_stats(source).colourful` ; `weak_bands` =
indices où `model.band_trust < 0.5` ; `low_colour = source_colourful < 0.03`.

**Interface.** Réutiliser le bandeau existant (`_mode_hint`, `_mode_hint_label`, ≈ l.
742–785) : en méthode `look`, afficher le texte de `SPEC.md` §4 correspondant, sans
bouton d'action. Si les deux limites sont vraies, afficher celle de la source.

**Tests.** `test_diagnostics_flag_a_grey_source` (image grise uniforme + bruit →
`low_colour`), `test_diagnostics_are_quiet_on_a_coloured_source`,
`test_look_banner_shows_low_colour` (GUI).

**Oracles.** Source du cas A : `source_colourful = 0,138` (± 0,005) → pas de bandeau.
Source de la paire n° 6 (`IMG_20260810_184343.jpg`) : 0,012 → bandeau.

**Ne pas faire.** Bloquer l'export ; modifier le rendu.

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

## T08 — Méthode par défaut · **décision de Mathieu**
Si Mathieu valide les planches : `Settings.color_transfer_method = "look"`
(`src/stealzalut/config.py`, deux occurrences) et mise à jour des tests qui supposent
`"mkl"` (chercher `color_transfer_method` et `mkl` dans `tests/`).

---

## Phase 2 — non planifiée
Sorties avec masques (une LUT par région + caches PNG, ou image gradée). Préalable :
une segmentation fiable (pinceau, ou modèle ONNX optionnel avec licence vérifiée) et une
planche qui montre un gain visible sur un cas réel. Ne pas commencer sans cela.
