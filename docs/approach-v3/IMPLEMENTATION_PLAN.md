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

## T09 — L'aperçu et le logiciel hôte ne voient pas la même image · **fait en partie**

**Constat (2026-11-12, paire de Mathieu : `IMG_6295.DNG` × `photo-1615723702352-…jpg`).**
Le `.cube` appliqué dans Lightroom ne donne pas les couleurs de l'aperçu. Mesuré :
- le fichier `.cube` relu et réappliqué à l'image de l'application donne **0 écart au
  pixel près** avec l'aperçu : l'export n'est pas en cause ;
- pour un DNG, l'application travaille sur le **JPEG intégré** (déjà développé par
  l'appareil) ; Lightroom applique la LUT sur **son propre développement** du brut.
  Entre le JPEG intégré et un développement LibRaw du même fichier : luminosité moyenne
  0,426 → 0,343, chroma moyenne 0,031 → 0,021. Une LUT est une fonction de la couleur
  d'entrée : entrée différente, résultat différent. **Ce n'est pas une limite du `.cube`.**
- le JPEG intégré est en **Display P3** et était lu comme du sRGB.

**Fait.**
- `utils/image_io.py` : `_pil_to_srgb`, `_has_wide_gamut_profile`. Toute image portant
  un profil ICC non sRGB (JPEG, HEIC, aperçu intégré d'un DNG) est convertie en sRGB au
  chargement. Les images sans profil ou en sRGB suivent le même chemin qu'avant.
- `core/lut_apply.py` : `apply_lut_full(image_rgb, grid, size)` (par bandes de 256
  lignes, même résultat que `apply_lut`).
- `MainWindow` : bandeau `_source_hint` quand la source est un fichier RAW ; bouton
  « Exporter l'image… » (`_on_export_image_clicked`) qui enregistre la source gradée en
  pleine résolution (JPEG qualité 95, TIFF ou PNG, 8 bits) avec la grille de l'état
  courant. Vérifié sur le DNG réel : 12 Mpx en 5,8 s, écart moyen avec l'aperçu de
  1 niveau sur 255 (redimensionnement et JPEG compris).
- Les réglages du mode Rendu (séparation, référence, recette) sont sur leur propre
  ligne : la barre du bas débordait à 1280 px de large.
- Tests : `tests/test_input_fidelity.py` (9), `TestSourceFidelity` (4) dans
  `tests/test_gui.py`.

**Non vérifié.**
- La conversion P3 → sRGB est testée par son câblage (la conversion est bien appelée),
  pas par une valeur de couleur : Pillow ne sait pas fabriquer un profil P3 pour un
  test. Mesure sur le DNG réel : chroma moyenne de la source 0,031 → 0,035.
- Que charger un export sRGB de Lightroom comme source donne bien, dans Lightroom,
  les couleurs de l'aperçu. Cela dépend de l'endroit où Lightroom applique la LUT d'un
  profil et de l'espace couleur choisi à la création du profil. **À vérifier par
  Mathieu avec un export réel** avant d'écrire quoi que ce soit de plus dans l'aide.

**Reste possible.** Développer soi-même le brut ne résout rien (aucun développement
tiers n'égale celui d'Adobe). L'export de l'image en 16 bits demanderait une source
16 bits, que l'application n'a pas (tout est en sRGB 8 bits).

**Effet sur les oracles.** `IMG_6295.DNG` étant en P3, ses mesures ont légèrement bougé
(source : 15,9 % de pixels colorés au lieu de 13,8 % ; cas A : chroma 0,046, pixels
colorés 29,5 % ; cas B : inchangé à 0,001 près). Tout reste dans les tolérances de
`ACCEPTANCE.md`. Pour T04, la part colorée de la source vaut maintenant 0,159.

---

## T10 — Espace couleur du `.cube` · **fait, accord avec les hôtes non vérifié**

**Principe.** Trois espaces distincts : celui de chaque image (lu dans son profil ICC,
converti à l'entrée) ; celui du moteur (Oklab, non borné) ; celui du `.cube`, choisi à
l'export, qui doit être celui où l'hôte l'applique (fenêtre de profil de Camera Raw,
espace de la timeline dans Resolve). C'est l'architecture habituelle d'une chaîne
gérée en couleur ; rien n'y est propre à StealZaLut.

**Fait.**
- `core/rgb_spaces.py` : `RGBSpace` (`to_oklab`, `from_oklab`, `oklab_to_linear`),
  `SRGB`, `DISPLAY_P3`, `ADOBE_RGB`, `PROPHOTO_RGB`, `REC709`, `CUBE_SPACES`,
  `space_by_key`. Les matrices sont **calculées** à partir des chromaticités publiées
  (ProPhoto, de blanc D50, est adapté à D65 par Bradford), pas recopiées.
- `colorspace.gamut_compress(lab, iters=12, to_linear=None)` : compression vers le
  gamut de n'importe quel espace.
- `TransferModel.apply_lab(lab) -> lab` : entrée Oklab, non bornée par sRGB. Par défaut,
  une couleur hors sRGB prend le déplacement de la couleur sRGB la plus proche de même
  teinte et luminosité ; `LookTransferModel` et `ZoneTransferModel` le font nativement.
- `build_grid(model, size, params, finishing, space=SRGB)` : pour `SRGB`, le chemin
  d'avant, **inchangé au bit près** (aperçu, oracles) ; sinon `_build_grid_in_space`.
- `ExportWorker(space=…)` ; menu « Espace du .cube : » (`cubeSpaceCombo`) ; le nom de
  fichier proposé porte l'espace (`…_display-p3.cube`) ; le message de fin d'export
  dit quoi choisir dans l'hôte.
- Tests : `tests/test_rgb_spaces.py` (44), `TestCubeSpace` (2) dans `tests/test_gui.py`.

**Preuve de cohérence.** `test_other_space_is_the_same_grade` : une image encodée dans
l'autre espace, passée par la grille de cet espace, puis ramenée en sRGB, donne le
rendu de l'aperçu (méthodes `look`, `zones`, `oklab` × 4 espaces). Sur la paire réelle
de Mathieu, à travers les fichiers `.cube` écrits par la vraie fenêtre : écart moyen
(distance Oklab) 0,0016 en Display P3, 0,0022 en Adobe RGB, 0,0038 en ProPhoto, 0,0021
en Rec.709 γ 2,4 ; 99ᵉ centile ≤ 0,012. L'essentiel vient de l'arrondi 8 bits.

**Non vérifié — à lire avant de s'y fier.**
- **Les courbes de transfert que chaque hôte associe à ces noms.** sRGB et Display P3
  (courbe sRGB) sont sûrs. Pour Adobe RGB (γ 563/256), ProPhoto (γ 1,8 avec talon
  linéaire) et Rec.709 (γ 2,4 pur), ce sont les définitions publiées ; je n'ai pas pu
  vérifier que Camera Raw applique bien celles-là à la LUT d'un profil. Un essai dans
  Lightroom avec une LUT neutre le dirait.
- **Les couleurs hors sRGB ne sont jamais vues à l'ajustement** : les images sont
  encore ramenées en sRGB 8 bits au chargement (aperçu, analyse). Dans un `.cube`
  large, ce que reçoivent ces couleurs est une extrapolation. Elle est lisse et ne
  replie pas, mais elle n'est validée par aucune image.
- La mesure de repliement suppose qu'augmenter R, V ou B augmente la luminosité : faux
  pour les primaires imaginaires de ProPhoto (sa grille neutre « replie » déjà de 0,6 %
  selon cette mesure). Le rendu `look` n'en ajoute pas.

**Reste possible.** Garder les images en espace large de bout en bout (lecture en
flottant, aperçu géré en couleur) : c'est ce qui lèverait la deuxième réserve.

---

## Phase 2 — non planifiée
Sorties avec masques (une LUT par région + caches PNG, ou image gradée). Préalable :
une segmentation fiable (pinceau, ou modèle ONNX optionnel avec licence vérifiée) et une
planche qui montre un gain visible sur un cas réel. Ne pas commencer sans cela.
