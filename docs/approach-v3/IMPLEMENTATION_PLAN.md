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

## T05 — Choisir ce que l'on lit dans la référence · **fait**
`zone_weights`, `FOCUS_RANGE` dans `core/look.py` ; `EditState.focus` ;
`TransferWorker(reference_weights=…)` ; dans `MainWindow` : bouton `focusButton`
(« Référence ▾ », visible en méthode `look`), `_open_focus_menu`, `_set_focus`,
`_clear_focus`, `_reference_zones`, `_reference_weights`, `_drop_focus`. La clé de cache
du modèle inclut le focus (`look:{focus}`) : changer un poids réajuste le modèle.
Tests : quatre `test_zone_weights_*` / `test_labels_match_the_analysis_copy` dans
`tests/test_look.py`, `TestLookFocus` (cinq) dans `tests/test_gui.py`.

**Écart par rapport au plan d'origine.** Le plan prévoyait un clic droit dans la vue des
zones ; cette vue est masquée hors de la méthode `zones`. Le choix se fait donc par un
bouton « Référence ▾ » placé après le curseur de séparation : un sous-menu par zone
(pastille de couleur, part de l'image) avec les trois choix de `SPEC.md` §4, et « Tout
remettre au poids normal ». Les zones sont celles de `analyze_zones(reference, 8)`,
calculées à la première ouverture du menu.

Mesuré sur le cas B (`prototypes/run_p4.py`, planche `out/p4_focus.jpg`) :

| Focus | L moy. | Chroma | Pixels colorés |
|---|---|---|---|
| aucun | 0,301 | 0,105 | 88,5 % |
| R1 (noirs, 17 % de l'image) ignorée | 0,326 | 0,123 | 96,3 % |
| R2 (ciel sombre, 31 %) ignorée | 0,325 | 0,110 | 74,9 % |
| R2 privilégiée | 0,295 | 0,106 | 93,0 % |

Reste connu : comme pour la séparation, un export lancé sans modèle en cache réajuste
sans les poids (`ExportWorker` n'a ni `prepare` ni `reference_weights`).

---

## T06 — Recette `.szl` : mode `look` · **fait**
`core/recipe.py` : `RECIPE_MODES` contient `"look"` ; `Recipe.separation`,
`Recipe.protect_vivid`, `Recipe.focus` ; `_parse_look` ; `look_focus(recipe,
reference_ids)` ; `look_recipe_to_dict(params, separation, protect_vivid, focus,
analysis_id, name, description, finishing)`. La version reste 2.

`gui/` : `EditState.protect_vivid: float | None = None` (`None` = la valeur choisie par
le modèle) ; `MainWindow.apply_recipe` aiguille vers `_apply_look_recipe` ;
`_export_look_recipe` ; `_look_analysis_id` ; `_prepare_model(tau, separation,
protect_vivid, model)`. Bouton « Recette ▾ » (`lookRecipeButton`), visible en méthode
`look`, avec « Coller la réponse de l'IA… », « Ouvrir une recette (.szl)… »,
« Enregistrer la recette… » (la barre des recettes existante est dans le panneau des
zones, masqué hors de la méthode `zones`).

Règles appliquées à l'import :
- `mode: "look"` → bascule sur la méthode `look` (comme `ambiance`) ;
- recette sans `mode`, importée en méthode `look` → ses réglages globaux et sa finition
  s'appliquent en `look` ;
- `mode: "zones"` importée en méthode `look` → message « Cette recette règle des
  correspondances entre zones : choisis la méthode « Zones » pour l'appliquer. » ;
- `analysis_id` différent de celui de la paire → tout s'applique **sauf** `focus`
  (les numéros de zones ne désignent plus les mêmes régions), et la barre d'état le dit ;
- zone de `focus` inconnue → recette refusée, rien n'est modifié ;
- `links` est ignoré en mode `look`, même mal formé.

`look.protect_vivid` n'a pas de curseur : c'est un réglage de recette. Le curseur
« Préserver les couleurs vives » des ajustements s'y ajoute.

Tests : `tests/test_look_recipe.py` (15 tests de format, dont
`test_v2_recipes_still_parse` sur la vraie réponse d'IA du cas A) et
`TestWindowLookRecipe` (7 tests) dans `tests/test_gui.py`. Vérifié dans la vraie
fenêtre sur le cas A : export, remise à zéro, réimport → même état, aperçu identique.

---

## T07 — Pack IA du mode `look` : l'IA voit le rendu · **fait, prompt non validé**
Nouveau module `src/stealzalut/core/look_pack.py` (le pack des zones, `core/ai_pack.py`,
n'est pas modifié) :

```python
LOOK_PACK_FILES = ("1_source.jpg", "2_reference.jpg", "3_current_render.jpg",
                   "4_reference_zones.png", "5_reference_overview.png",
                   "analysis.json", "PROMPT.md")
def look_gaps(render_rgb, reference_rgb) -> dict[str, Any]
def gaps_table(gaps: dict[str, Any]) -> str
def look_analysis_dict(reference_rgb, reference: ZonePalette, analysis_id: str,
                       separation: float, protect_vivid: float,
                       focus: tuple[tuple[int, float], ...], gaps) -> dict[str, Any]
def example_recipe(analysis_id: str, first_zone: int) -> dict[str, Any]
def build_look_prompt(reference: ZonePalette, analysis_id: str, separation: float,
                      protect_vivid: float, focus, gaps) -> str
def write_look_pack(destination: Path, name: str, source_rgb, reference_rgb, render_rgb,
                    reference: ZonePalette, analysis_id: str, separation: float,
                    protect_vivid: float, focus=()) -> tuple[Path, str]
```

Le dossier s'appelle `<nom>_stealzalut_look_pack` (un `.zip` à côté). Le prompt est le
texte de `_PROMPT` dans ce module : **c'est lui qui fait foi**, il remplace le brouillon
qui figurait ici. Il contient un tableau d'écarts rendu / référence : luminosité
moyenne, chroma moyenne, part de pixels colorés, et la couleur moyenne (en hexadécimal)
des ombres (L < 0,35), des tons moyens et des hautes lumières (L ≥ 0,65), mesurés sur
des vignettes de 640 px.

Interface : `MainWindow._on_prepare_look_pack`, première entrée du menu « Recette ▾ »
(« Préparer pour mon IA… »). Le prompt est copié dans le presse-papiers. La protection
des couleurs vives annoncée à l'IA est la valeur effective du modèle.

Écarts par rapport au plan d'origine : module séparé plutôt que des paramètres ajoutés à
`write_pack` (le mode `look` n'a pas de `ZoneTransferModel`) ; le rendu est le fichier
n° 3, pas n° 7 ; les planches de zones de la source ne sont pas dans le pack (le mode
`look` ne s'en sert pas).

Tests : `tests/test_look_pack.py` (8 tests, dont `test_look_prompt_example_parses` :
l'exemple montré à l'IA est lui-même une recette valide) et `TestWindowLookPack`
(2 tests) dans `tests/test_gui.py`. Vérifié dans la vraie fenêtre : packs écrits pour
les cas A et B en moins d'une seconde.

**Reste à faire, et ce n'est pas optionnel.** Ce prompt n'a été soumis à aucune IA.
Donner un vrai pack (cas A et B) à l'IA de Mathieu, importer sa réponse, joindre la
planche avant / après à `ACCEPTANCE.md`, et corriger le prompt selon ce que l'IA en
fait réellement (voir le piège n° 8 du brief : un conseil de prompt peut faire plus de
dégâts qu'un bug). Points à surveiller : l'IA recopie-t-elle les valeurs de l'exemple ?
Utilise-t-elle les roues avec la bonne échelle de teintes ? Ignore-t-elle des zones à
tort ?

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
