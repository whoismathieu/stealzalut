# Spécification v3 — méthode « Rendu » (`look`)

> Pour un exécutant qui n'a que ce dépôt et ces fichiers. Lis d'abord `DESIGN.md` (le
> pourquoi), puis ce fichier (le quoi), puis `IMPLEMENTATION_PLAN.md` (l'ordre).
> **Tout symbole cité « existant » a été vérifié au commit `098396b`** ; s'il a changé,
> arrête-toi et relis le fichier au lieu de deviner.

## 0. Décidé / ouvert / interdit

**Décidé (ne pas rediscuter)**
- La méthode `look` est un `TransferModel` de plus. `zones`, `ambiance` et les méthodes
  globales restent, inchangées.
- La luminosité de `look` ne dépend que de la luminosité d'entrée (courbe monotone).
  Aucune tâche ne doit introduire de dépendance de `L'` à la couleur.
- La couleur est apprise par bande de tons, conditionnellement au ton de sortie.
- Aucune nouvelle dépendance (ni SciPy, ni onnxruntime). numpy + OpenCV suffisent.
- `look` est la méthode par défaut (décision de Mathieu du 2026-11-11).
- Hors ligne, aucune API. Interface en français, code en anglais.
- Les recettes `.szl` version 2 existantes restent lisibles sans changement.

**Ouvert (décision de Mathieu, ne pas trancher seul)**
- Retirer ou garder `ambiance` une fois `look` validé visuellement.
- Phase 2 : sorties avec masques (plusieurs LUT + caches, image gradée).
- Un réglage « densité » (couleurs saturées plus sombres) : non prototypé.

**Interdit**
- Mentionner un modèle d'IA comme auteur où que ce soit (commits, fichiers).
- `git push`, réécriture d'historique, affichage de la configuration du dépôt distant.
- Modifier les constantes de §2 sans rejouer `prototypes/run_p1.py` et mettre à jour
  `ACCEPTANCE.md`.
- Ajouter un lissage automatique ou un contrôle de repliement propre à `look` : il n'en
  a pas besoin.
- Écrire des fichiers volumineux dans le dépôt (disque presque plein).

## 1. Symboles existants utilisés (vérifiés au commit `098396b`)

| Symbole | Fichier | Signature copiée |
|---|---|---|
| `TransferModel` | `core/color_transfer.py` | classe abstraite ; `method` (propriété, `str`) ; `apply(self, colors: NDArray[np.float64]) -> NDArray[np.float64]` ; `fold_reference(self) -> TransferModel` ; `bake(self, size: int)` |
| `fit_transfer` | `core/color_transfer.py` | `fit_transfer(source: NDArray[np.uint8], reference: NDArray[np.uint8], method: str = "mkl") -> TransferModel` |
| `SUPPORTED_METHODS`, `METHOD_LABELS`, `COLOR_MATCHER_METHODS` | `core/color_transfer.py` | `tuple[str, ...]` (11 entrées, `"look"` en tête), `dict[str, str]`, `frozenset[str]` |
| `srgb_to_oklab`, `oklab_to_srgb`, `gamut_compress` | `core/colorspace.py` | `(rgb: F64) -> F64`, `(lab: F64) -> F64`, `gamut_compress(lab: F64, iters: int = 12) -> F64` |
| `eval_curve` | `core/finishing.py` | `eval_curve(points: tuple[tuple[float, float], ...], x: F64) -> F64` |
| `Finishing` | `core/finishing.py` | dataclass : `shadows`, `midtones`, `highlights` (`Wheel`), `tone_curve`, `bands`, `saturation`, `vibrance` |
| `GradingParams` | `core/grading.py` | dataclass : `luminance_strength=1.0`, `chroma_strength=1.0`, `preserve_neutrals=0.0`, `skin_protection=0.0`, `preserve_vivid=0.0`, `mix=1.0`, `smoothing=0.0` |
| `build_grid` | `core/grading.py` | `build_grid(model: TransferModel, size: int, params: GradingParams \| None = None, finishing: Finishing \| None = None) -> GridResult` |
| `GridResult` | `core/grading.py` | `grid`, `fold_fraction`, `smoothing_used`, `auto_smoothed` |
| `apply_lut` | `core/lut_apply.py` | `apply_lut(image_rgb: NDArray[np.uint8], grid: NDArray[np.float64], size: int) -> NDArray[np.uint8]` |
| `ANALYSIS_MAX_SIZE`, `colour_stats`, `ColourStats`, `ZonePalette` | `core/palette.py` | `256` ; `colour_stats(image_rgb) -> ColourStats` (`concentration`, `colourful`, `hue`, `is_monochrome`) ; `ZonePalette.labels` : `(h, w)` int32, identifiant de zone (1…K) de chaque pixel **de la copie d'analyse** |
| `create_thumbnail` | `utils/image_io.py` | `create_thumbnail(image: ImageArray, max_size: int = 512) -> ImageArray` |
| `Recipe`, `parse_recipe`, `apply_global`, `recipe_to_dict`, `RECIPE_MODES`, `RecipeError` | `core/recipe.py` | `Recipe` : `analysis_id`, `name`, `description`, `grading: dict[str, float]`, `tau`, `zone_count`, `links`, `finishing`, `finishing_keys`, `mode` ; `parse_recipe(source: str \| dict[str, Any]) -> Recipe` ; `apply_global(recipe, params, tau) -> tuple[GradingParams, float]` ; `RECIPE_MODES = ("zones", "ambiance")` |
| `write_pack`, `build_prompt`, `PACK_FILES`, `_PROMPT` | `core/ai_pack.py` | `write_pack(destination: Path, name: str, source_rgb, reference_rgb, model: ZoneTransferModel, zone_count: int) -> Path` ; `build_prompt(model: ZoneTransferModel) -> str` |
| `EditState`, `History.push` | `gui/history.py` | `EditState(table: CorrespondenceTable \| None, params: GradingParams, tau: float, finishing: Finishing = Finishing(), separation: float = 1.0)` ; `push(self, state: EditState, merge_key: str \| None = None) -> bool` |
| `MainWindow._effective_model`, `_prepare_model`, `_run_transfer_preview`, `_tau_slider`, `_separation_slider` | `gui/main_window.py` | cherche-les par leur nom (les numéros de ligne bougent) et lis-les avant de les toucher |
| `TransferWorker` | `workers/transfer_worker.py` | paramètre `prepare: Callable[[TransferModel], TransferModel] \| None` appliqué après l'ajustement |

## 2. Module livré : `src/stealzalut/core/look.py`

API publique (déjà implémentée, ne pas changer les signatures) :

```python
def weighted_quantile(values: F64, weights: F64, u: F64) -> F64
def fit_tone_curve(source_l: F64, reference_l: F64,
                   reference_weights: F64 | None = None) -> tuple[F64, F64]
def sliced_transport(source: F64, target: F64, slices: int = SLICES) -> F64
def lattice_position(ab: F64) -> F64

@dataclass(frozen=True, slots=True, eq=False)
class LookTransferModel(TransferModel):
    tone_x: F64            # (129,)
    tone_y: F64            # (129,) non décroissant
    band_lightness: F64    # (12,) croissant
    cast: F64              # (12, 2)
    field: F64             # (12, 49, 49, 2)
    band_trust: F64        # (12,) dans [0, 1]
    separation: float = 1.0
    protect_vivid: float = 0.0
    def with_separation(self, separation: float) -> LookTransferModel   # borné à [0, 1.5]
    def with_protect_vivid(self, protect_vivid: float) -> LookTransferModel  # borné à [0, 1]
    def tone(self, lightness: F64) -> F64
    def chroma_shift(self, lab: F64) -> F64
    def apply(self, colors: F64) -> F64

def fit_look(source: NDArray[np.uint8], reference: NDArray[np.uint8],
             reference_weights: F64 | None = None) -> LookTransferModel
```

`reference_weights` : un poids ≥ 0 par pixel de `create_thumbnail(reference, 256)`,
aplati ligne par ligne. Mauvaise longueur, poids négatif ou somme nulle → `ValueError`
dont le message contient `reference_weights`.

### Constantes

| Nom | Valeur | Unité | Rôle | Réglage |
|---|---|---|---|---|
| `TONE_QUANTILES` | 0,01 ; 0,05 ; 0,15 ; 0,30 ; 0,50 ; 0,70 ; 0,85 | quantile | masse de l'image | 0,85 choisi pour exclure les enseignes du cas A |
| `WHITE_QUANTILE` | 0,999 | quantile | point blanc | essai unique |
| `TONE_SLOPE` | (0,35 ; 3,0) | pente | bornes de contraste | essai unique |
| `TONE_MIN_GAP` | 0,01 | L | nœuds trop proches ignorés | — |
| `TONE_SAMPLES` | 129 | — | échantillons de la courbe | — |
| `BANDS` | 12 | — | bandes de tons | 12 ≈ 8 % des pixels par bande |
| `BAND_SAMPLES` | 3000 | points | tirage par bande | compromis bruit / durée (0,3 s) |
| `BAND_MIN_WIDTH` | 0,02 | L | largeur minimale d'une bande | — |
| `MIN_BAND_SHARE` | 0,004 | part | seuil de confiance | essai unique |
| `SLICES` | 48 | — | directions du transport | convergence visuelle en 2D |
| `BAND_SMOOTHING` | 0,7 | bande | lissage entre bandes | — |
| `COMPAND` | 0,05 | chroma | finesse du treillis près du gris | sans compandage la séparation disparaissait |
| `AB_CELLS` | 49 | nœuds | treillis de chroma | — |
| `SIGMA_FINE` | 1,5 | maille | lissage du résidu | 0,75 testé : pas de gain (`diag_loss.py`) |
| `SIGMA_COARSE` | 6,0 | maille | champ lointain | — |
| `FIELD_PRIOR` | 2,0 | points/maille | poids du champ lointain | — |
| `FAR_PRIOR` | 0,05 | points/maille | retour à zéro loin des données | — |
| `VIVID_CHROMA` | (0,04 ; 0,12) | chroma | seuil « couleur vive » | identique à `grading.VIVID_CHROMA` |
| `MONOCHROME_VIVID_PROTECTION` | 0,8 | — | protection si référence monochrome | repris du mode ambiance |
| `SEPARATION_RANGE` | (0,0 ; 1,5) | — | bornes du réglage | 1,5 : +3 points de pixels colorés, sans artefact vu |

Toutes sont des **valeurs d'essai** réglées sur huit paires urbaines.

### Pseudo-code de `fit_look`
```
lab_s, lab_r = Oklab des copies 256 px
T = fit_tone_curve(L_s, L_r, poids)
ℓ_m = quantile(L_s, (m + 0.5)/12), m = 0…11 ; rendu strictement croissant
ℓ'_m = T(ℓ_m) ; largeurs = max(gradient, 0.02)
pour chaque bande m :
    w_s = exp(-½((L_s − ℓ_m)/largeur_s)²) ; w_r = poids · exp(-½((L_r − ℓ'_m)/largeur_r)²)
    confiance = min(1, (Σw_r/Σpoids)/0.004)²
    P = tirage_systématique(ab_s, w_s, 3000) ; Q = tirage_systématique(ab_r, w_r, 3000)
    D = confiance · (transport_tranché(P, Q) − P)
    dominante_m = moyenne(D)
    (somme, compte) = dépôt bilinéaire de (D − dominante_m) aux positions treillis(P)
    lointain = flou(somme, 6)/(flou(compte, 6) + 0.05)
    champ_m = (flou(somme, 1.5) + 2·lointain)/(flou(compte, 1.5) + 2)
lisser dominantes et champs entre bandes (σ = 0.7)
protection = 0.8 si colour_stats(reference).is_monochrome sinon 0
```

### Cas limites
- Image uniforme : les quantiles sont confondus, les nœuds trop proches sont ignorés, la
  courbe reste monotone ; ne pas lever d'exception.
- Référence sans pixel à un ton : `confiance` → 0, la bande ne déplace rien.
- Couleurs de la grille absentes de la source : champ lointain, borné, lisse.

### Budget de performance (mesuré sur la machine de Mathieu)
Ajustement ≤ 0,5 s ; cuisson 33³ ≤ 0,3 s ; 64³ ≤ 0,8 s.

## 3. Modules à créer ou étendre (voir le plan pour l'ordre)

### 3.1. `core/look.py` — diagnostics (T04)
```python
@dataclass(frozen=True, slots=True)
class LookDiagnostics:
    source_colourful: float      # ColourStats.colourful de la source
    weak_bands: tuple[int, ...]  # indices m avec band_trust[m] < 0.5
    low_colour: bool             # source_colourful < LOW_COLOUR_SHARE

LOW_COLOUR_SHARE = 0.03  # paire n° 6 (source grise) : 0.012 ; cas A : 0.138

def look_diagnostics(model: LookTransferModel, source: NDArray[np.uint8]) -> LookDiagnostics
```

### 3.2. `core/look.py` — poids par zone (T05)
```python
def zone_weights(labels: NDArray[np.int32], focus: Mapping[int, float]) -> F64
```
Retourne `labels.size` poids : `focus.get(id, 1.0)` pour chaque pixel, aplati ligne par
ligne. Poids hors de [0 ; 4] → `ValueError`. `labels` vient de
`ZoneTransferModel.reference.labels` (même copie d'analyse que `fit_look` : vérifier par
un test que `labels.size == create_thumbnail(reference, 256).shape[0] * shape[1]`).

### 3.3. `gui/history.py` — état (T03)
`EditState` a déjà `separation: float = 1.0` après `finishing` (T03, fait). Ajouter à la suite :
```python
focus: tuple[tuple[int, float], ...] = ()   # (zone de référence, poids), T05
```

### 3.4. `core/recipe.py` — mode `look` (T06)
- `RECIPE_MODES = ("zones", "ambiance", "look")`.
- `Recipe` gagne `separation: float | None = None`, `protect_vivid: float | None = None`,
  `focus: dict[int, float] = field(default_factory=dict)`.
- Bloc JSON accepté, facultatif, uniquement si `mode == "look"` :
```json
"look": { "separation": 1.2, "protect_vivid": 0.5,
          "focus": { "R1": 0.0, "R5": 2.0 } }
```
  `separation` ∈ [0 ; 1,5], `protect_vivid` ∈ [0 ; 1], poids de `focus` ∈ [0 ; 4], clés
  `"R<n>"` ou entiers. `links` est ignoré en mode `look` (ne pas lever d'erreur).
- Messages d'erreur (français, mot pour mot) :
  - `"« look.separation » doit être un nombre entre 0 et 1.5."`
  - `"« look.protect_vivid » doit être un nombre entre 0 et 1."`
  - `"« look.focus » : le poids de {clé} doit être un nombre entre 0 et 4."`
  - `"« look.focus » : « {clé} » n'est pas une zone de la référence (R1 à R{n})."`

### 3.5. `core/ai_pack.py` — pack du mode `look` (T07)
- Nouveau fichier de pack `7_current_render.jpg` (rendu actuel, ≤ 1600 px, qualité 90).
- `analysis.json` gagne une clé `"look"` :
```json
"look": { "separation": 1.0, "protect_vivid": 0.0,
  "gaps": { "lightness_mean": [0.296, 0.280], "chroma_mean": [0.045, 0.053],
            "colourful_share": [0.282, 0.434],
            "shadows_ab": [[-0.031, 0.015], [-0.032, 0.015]] } }
```
  Chaque paire est `[rendu, référence]`, calculée sur des vignettes de 640 px ;
  « ombres » = pixels de L < 0,35 ; « colorés » = C ≥ 0,05.
- Texte du prompt du mode `look` : voir `IMPLEMENTATION_PLAN.md`, T07 (mot pour mot).

## 4. Textes d'interface (mot pour mot)

| Où | Texte |
|---|---|
| Liste des méthodes (livré) | `Rendu (tons et couleurs de la référence)` |
| Libellé du curseur (livré) | `Séparation :` |
| Infobulle du curseur (livré) | `Écarte les couleurs proches du gris vers les couleurs de la référence.\n0 : seulement la dominante de chaque ton. 1 : ce que montre la référence. Au-delà : accentué.` |
| Bandeau, source peu colorée (T04) | `La photo a peu de couleurs : la séparation aura peu d'effet.` |
| Bandeau, tons absents (T04) | `La référence ne montre presque rien dans certains tons : ils gardent leurs couleurs.` |
| Menu des zones de référence (T05) | `Ignorer cette zone pour le rendu` / `Privilégier cette zone pour le rendu` / `Poids normal` |

## 5. Glossaire

- **Oklab** : espace couleur perceptuel ; `L` luminosité, `(a, b)` couleur ; chroma `C = √(a² + b²)`.
- **Ton** : luminosité `L` d'un pixel. **Bande de tons** : tranche de `L` (12 dans `look`).
- **Dominante** : déplacement moyen de `(a, b)` dans une bande.
- **Résidu / séparation** : ce que le transport ajoute à la dominante ; son poids.
- **Transport tranché** : appariement de deux nuages de points par projections successives.
- **Compandage** : changement d'échelle de la chroma qui resserre les mailles près du gris.
- **Zone** : grappe de couleurs (k-means) d'une image. **Lien / table** : correspondances
  zone source → zones de référence (méthode `zones`).
- **τ (tau)** : netteté des transitions entre zones (méthode `zones` seulement).
- **Recette** : fichier `.szl`, bloc JSON de réglages. **Pack** : dossier remis à l'IA.
- **Repliement** : la luminosité de sortie décroît quand celle d'entrée croît ;
  mesuré par `fold_fraction`.
- **Copie d'analyse** : image réduite à 256 px sur son grand côté.
