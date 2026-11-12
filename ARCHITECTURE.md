# StealZaLut — Architecture Technique & Contexte de Référence

> **Source de vérité unique.** Ce document et `README.md` constituent les **seuls fichiers de contexte principaux** du projet. Ils remplacent et consolident l'ensemble des notes historiques (`CONTEXT.md`, `docs/approach-v3/*`, `docs/audit/*`).

---

## 1. Règles d'Ingénierie & Garde-Fous (Non-Négociables)

Tout développeur ou assistant IA intervenant sur ce dépôt doit impérativement respecter ces règles :

1. **Unicité documentaire (Anti-Bloat)** : Ne jamais créer de fichiers `.md` secondaires (journaux d'état, sous-plans, notes de session) pour itérer. Toute mise à jour architecturale s'effectue uniquement dans `ARCHITECTURE.md` et `README.md`.
2. **Zéro attribution IA** : N'écrire sous aucun prétexte `Co-Authored-By`, `Generated with…`, ni le nom d'un modèle d'IA dans les commits, commentaires, docstrings ou fichiers du dépôt.
3. **Sécurité Git & Réseau** :
   * Ne **jamais** exécuter `git push`, `git remote -v`, `git config --list` ni lire `.git/config` (l'URL distante contient un jeton en clair).
   * L'application est **100 % hors-ligne** : aucune dépendance réseau, aucun appel d'API IA, aucune clé.
4. **Économie d'espace disque** : La partition locale étant saturée ($> 95\,\%$), ne jamais écrire d'images de test, de fichiers `.npy` ou de planches comparatives dans le dépôt. Utiliser exclusivement un dossier temporaire hors dépôt (`mktemp -d`) et le supprimer immédiatement après usage.
5. **Conventions de langue & validation** :
   * Textes d'interface (UI) et messages d'erreur utilisateur en **français** ; code, identifiants, docstrings et messages de commit (`feat:`, `fix:`, `docs:`) en **anglais**.
   * Avant chaque commit, les trois commandes suivantes doivent réussir sans erreur ni avertissement :
     ```bash
     .venv/Scripts/python.exe -m pytest -q
     .venv/Scripts/python.exe -m ruff check src tests
     .venv/Scripts/python.exe -m mypy src
     ```

---

## 2. Découplage Architectural & Arborescence

L'application suit un patron **Model $\to$ Workers $\to$ View** strictement découplé :
* **`src/stealzalut/core/`** : Mathématiques pures, science des couleurs et sérialisation. **Interdiction absolue d'y importer `PySide6`.**
* **`src/stealzalut/workers/`** : Encapsulation des calculs dans des `QThread` (`PySide6.QtCore` uniquement), communiquant vers l'UI par `Signal`.
* **`src/stealzalut/gui/`** : Couche de présentation `PySide6.QtWidgets` et historique d'annulation immuable.

```text
src/stealzalut/
├── main.py                     # Point d'entrée QApplication + chargement theme.qss
├── config.py                   # Chargement .env -> dataclass gelée Settings (défaut: method="look")
├── core/                       # ★ LOGIQUE PURE (Zéro dépendance Qt)
│   ├── colorspace.py           # Conversions sRGB <-> RGB linéaire <-> Oklab <-> Oklch, gamut_compress
│   ├── rgb_spaces.py           # RGBSpace (Bradford D50->D65), CUBE_SPACES (sRGB, P3, AdobeRGB, Rec709)
│   ├── color_transfer.py       # Classe abstraite TransferModel, fit_transfer, SUPPORTED_METHODS (11)
│   ├── look.py                 # Moteur par défaut "look" (courbe monotone + transport chroma par bande)
│   ├── palette.py              # Segmentation k-means Oklab déterministe (256 px), ColourStats, analysis_id
│   ├── matching.py             # Appariement optimal entropique de Sinkhorn entre ZonePalettes
│   ├── correspondence.py       # CorrespondenceTable immuable, Link, Intent, opérations d'édition pures
│   ├── zone_transfer.py        # Moteur "zones", mélange doux de Mahalanobis, conflits de Bhattacharyya
│   ├── ambiance.py             # Moteur "ambiance" (courbe de teinte par L pour références monochromes)
│   ├── grading.py              # GradingParams, masques (neutres/peau/vifs), build_grid unifié
│   ├── regularize.py           # Flou gaussien séparable 3D (smooth_field), mesure de repli (fold_fraction)
│   ├── finishing.py            # Finishing : 3 roues de Bernstein, courbe PCHIP, 8 bandes TSL, sat/vibrance
│   ├── lut_engine.py           # Grille neutre [0,1]^3 (R rapide, G moyen, B lent), VALID_LUT_SIZES={17,33,64}
│   ├── lut_apply.py            # Interpolation trilinéaire (apply_lut, apply_lut_colors, render_lut)
│   ├── lut_export.py           # Écriture/lecture du format standard Iridas .cube
│   ├── recipe.py               # Parseur et sérialiseur de recettes .szl v2 (modes look, zones, ambiance)
│   ├── look_pack.py            # Pack IA du mode "look" (inclut 3_current_render.jpg et tableau d'écarts)
│   └── ai_pack.py              # Pack IA du mode "zones" (planches de zones isolées et vues d'ensemble)
├── workers/                    # ★ THREADS ASYNCHRONES (QtCore)
│   ├── transfer_worker.py      # Calcul du modèle, cuisson de la grille d'aperçu et rendu WideImage
│   └── export_worker.py        # Cuisson à la résolution cible et écriture du fichier .cube
├── gui/                        # ★ PRÉSENTATION (QtWidgets)
│   ├── main_window.py          # Orchestration UI, cache du modèle (_model_cache), gestion workers périmés
│   ├── history.py              # Pile Undo/Redo immuable (EditState avec fenêtre de fusion 1.0 s)
│   ├── overlay.py              # Surlignage et détourage des zones sur les vignettes
│   ├── zone_text.py            # Libellés français des intentions, origines et couleurs dominantes
│   ├── styles/theme.qss        # Thème sombre Qt (polices en px)
│   └── widgets/                # image_drop_zone, preview_panel, correspondence_view, zone_inspector,
│                               # adjustments_panel, finishing_panel, paste_recipe_dialog
├── utils/
│   ├── image_io.py             # WideImage, décodage ICC natif, RAW/DNG (aperçu intégré / rawpy), HEIC
│   └── paths.py                # Résolution des chemins compatible PyInstaller (sys._MEIPASS)
└── devtools/
    ├── ground_truth.py         # Banc de mesure à vérité connue en Display P3 (non importé par l'appli)
    └── smoke_gui.py            # Capture de la vraie fenêtre en mode "look" sur une paire (contrôle visuel)
```

---

## 3. Pipeline Colorimétrique Large Gamut (`WideImage` $\to$ Oklab $\to$ `.cube`)

### 3.1. Les Trois Espaces de Couleur
1. **Espace de lecture (`utils/image_io.py:WideImage`)** :
   * `load_image_rgb` retourne une instance de `WideImage` (sous-classe de `np.ndarray` `uint8`).
   * Le tableau `uint8` lui-même contient la réduction sRGB 8 bits destinée uniquement à l'affichage Qt.
   * Lorsque le fichier source possède un profil ICC large gamut reconnu par sous-chaîne (`Display P3`, `Adobe RGB` / `ClayRGB` / `opRGB`, `ProPhoto` / `ROMM`), `WideImage` transporte deux attributs critiques :
     * `analysis_lab` : tableau `(h, w, 3)` en `float64` contenant les couleurs **Oklab non écrêtées** de la copie d'analyse ($256\text{ px}$), calculées directement depuis les valeurs du fichier via les matrices exactes de `core/rgb_spaces.py`.
     * `native` et `native_space` : les pixels pleine résolution tels qu'encodés dans le fichier et leur objet `RGBSpace`.
   * **Invariant `WideImage`** : Tout découpage, copie ou redimensionnement (`create_thumbnail`, `np.asarray`, slicing) déclenche `__array_finalize__` et remet `analysis_lab` et `native` à `None`. Le point d'entrée unique du moteur pour lire l'analyse est `core/palette.py:analysis_oklab(image)`.
2. **Espace du moteur (`core/colorspace.py`)** :
   * Tout le moteur travaille en **Oklab non borné**. La transformation linéaire vers LMS utilise une racine cubique signée (`np.cbrt(lms)`), préservant sans écrêtage les coordonnées hors gamut sRGB.
   * Chaque modèle Oklab (`LookTransferModel`, `ZoneTransferModel`, `AmbianceTransferModel`, `OklabTransferModel`) implémente nativement `apply_lab(self, lab: F64) -> F64` sans passer par sRGB.
3. **Espace d'export et d'application de la LUT (`core/rgb_spaces.py` & `core/lut_apply.py`)** :
   * Espaces disponibles dans `CUBE_SPACES` : `SRGB`, `DISPLAY_P3`, `ADOBE_RGB` ($\gamma = 563/256$), `REC709` ($\gamma = 2,4$).
   * `PROPHOTO_RGB` (adapté de D50 vers D65 par matrice de Bradford) est défini pour la lecture des fichiers mais **strictement exclu de `CUBE_SPACES`** : ses primaires imaginaires placent les sommets de $[0, 1]^3$ dans des régions où les réponses LMS sont négatives, détruisant la monotonie de $L$ ($0,6\,\%$ de repliement dès la grille identité).
   * Dans `render_lut`, les pixels `native` de la source sont convertis vers l'espace choisi pour le `.cube`, interpolés trilinéairement dans la grille 3D de cet espace, puis convertis et compressés en gamut vers sRGB 8 bits uniquement à la sortie pour l'écran ou l'export d'image.

### 3.2. Chaîne de Cuisson Unifiée (`core/grading.py:build_grid`)
Pour une résolution $N \in \{17, 33, 64\}$ et un espace cible `space: RGBSpace`, la séquence est unique :
1. **Entrée** : Grille neutre $[0, 1]^3$ de forme $(N^3, 3)$ convertie en Oklab via $\text{lab}_{\text{in}} = \text{space.to\_oklab}(\text{grid})$.
2. **Transfert brut** : $\text{lab}_{\text{raw}} = \text{model.apply\_lab}(\text{lab}_{\text{in}})$.
3. **Dosage & Masques (`adjust_displacement`)** : Sur le champ de déplacement $\Delta = \text{lab}_{\text{raw}} - \text{lab}_{\text{in}}$ :
   * Échelle séparée de $\Delta L$ (`luminance_strength`) et $\Delta ab$ (`chroma_strength`).
   * Atténuation multiplicative de $\Delta ab$ uniquement (ne touche jamais $\Delta L$ pour éviter tout repliement) par trois masques `smoothstep` évalués en Oklch sur l'entrée :
     * `neutral_mask` : protège les gris ($C \in [0,02, 0,06]$).
     * `skin_mask` : fenêtre sur $h \in [27,5^\circ, 87,5^\circ]$ (centre $57,5^\circ$), $C \in [0,015, 0,20]$, $L \in [0,25, 0,98]$.
     * `vivid_mask` : protège les couleurs saturées ($C \in [0,04, 0,12]$).
   * Multiplication globale par `mix`.
4. **Régularisation & Contrôle de Repliement (`core/regularize.py`)** :
   * Flou gaussien séparable 3D `smooth_field` de rayon $\sigma_{\text{cells}} = \text{smoothing} \times \frac{N - 1}{32}$.
   * Mesure de `fold_fraction` (part des arêtes de la grille 3D le long desquelles $L$ décroît de plus de $10^{-3}$ quand $R$, $G$ ou $B$ croît). Si les réglages ajoutent plus de $0,5\,\%$ (`MAX_EXTRA_FOLDS`) de replis par rapport à `model.fold_reference()`, le lissage est automatiquement doublé (départ à $0,5$, jusqu'à $3$ étapes).
5. **Finition (`core/finishing.py:apply_finishing`)** :
   * Courbe tonale cubique monotone (Fritsch–Carlson / PCHIP sans dépendance SciPy) sur $L$.
   * 3 roues colorimétriques (Ombres, Tons moyens, Hautes lumières) pondérées par la base de Bernstein $(1-L)^2$, $2L(1-L)$, $L^2$.
   * 8 bandes de couleurs gaussiennes ($\sigma = 22^\circ$, pondérées par `smoothstep(C, 0.01, 0.04)` pour ignorer les gris) + Saturation et Vibrance globales.
6. **Sortie & Gamut** : Compression de gamut à teinte et luminosité constantes par recherche dichotomique (12 itérations) vers le cube linéaire de l'espace cible (`gamut_compress(lab_out, to_linear=space.oklab_to_linear)`), puis encodage `space.from_oklab`.

---

## 4. Mathématiques des Moteurs de Transfert

### 4.1. Moteur `look` (« Rendu » — `core/look.py`)
Conçu pour reproduire le comportement d'un tirage argentique sans créer d'inversions tonales ni perdre de saturation au passage du gamut :

1. **Courbe de tons monotone $T(L)$** :
   * Apparie les quantiles de luminosité source et référence (pondérés par `reference_weights`) aux niveaux $u \in \{0,01, 0,05, 0,15, 0,30, 0,50, 0,70, 0,85\}$ (la masse de la scène) et $u = 0,999$ (le point blanc).
   * Le saut de $85\,\%$ à $99,9\,\%$ est critique : il aligne le niveau des sources lumineuses (enseignes, néons) sans les forcer à occuper la même surface que dans la référence.
   * La pente entre deux nœuds consécutifs est bornée dans $[0,35, 3,0]$ (`TONE_SLOPE`), puis interpolée par cubique monotone sur $129$ échantillons.
2. **Transport de chroma conditionnel au ton ($12$ bandes)** :
   * Pour chaque bande $m \in \{0, \dots, 11\}$ centrée sur le quantile source $\ell_m$ et envoyée au ton de sortie $\ell'_m = T(\ell_m)$, on tire systématiquement $3\,000$ points $(a, b)$ dans une fenêtre gaussienne autour de $\ell_m$ (source) et $\ell'_m$ (référence).
   * Le nuage source $P$ est transporté sur le nuage cible $Q$ par **transport optimal tranché** (`sliced_transport`, $48$ directions suivant la suite du nombre d'or), pondéré par la confiance $\text{trust}_m = \min\left(1, \frac{\text{part}_m}{0,004}\right)^2$.
   * Le déplacement $D = \text{trust}_m \cdot (\text{moved} - P)$ est décomposé en trois termes :
     $$\Delta ab(L, a, b) = \text{cast}(L) + \text{separation} \cdot \Big( (ab - c(L))\,\text{stretch}(L)^\top + \text{field}(L, p_a, p_b) \Big)$$
     * **$\text{cast}_m$** : déplacement moyen $\bar{D}$ de la bande (la dominante).
     * **$\text{stretch}_m$** : application linéaire $2 \times 2$ ajustée par moindres carrés régularisés (*ridge*, `STRETCH_PRIOR = 0.005`) sur $D - \bar{D}$, dont les valeurs singulières de $I + B$ sont bornées dans $[0,5, 2,0]$. Permet de transporter exactement un gain de saturation ou une rotation de teinte même hors du support de la source.
     * **$\text{field}_m$** : résidu non linéaire déposé bilinéairement sur un treillis $49 \times 49$ en coordonnées compandées $p = \frac{ab}{\|ab\| + 0,05}$, combinant un flou fin ($\sigma = 1,5\text{ maille}$) et un champ lointain ($\sigma = 6,0\text{ mailles}$, poids $2,0$).
3. **Protection sélective par teinte (`protect_hue`)** :
   * Si la référence est détectée comme monochrome (`concentration >= 0.85`), `protect_vivid` vaut automatiquement $0,8$, mais uniquement pour les teintes de la source distantes de plus de $15^\circ\text{ à }45^\circ$ (`PROTECT_HUE_RANGE`) de la teinte dominante de la référence.
4. **Localité / Fidélité à ma photo (`locality` $\in [0, 1]$)** :
   * Convertit le curseur $t \in (0, 1]$ en une portée maximale $r = 0,30 \cdot (0,02 / 0,30)^t$ pour la chroma et $r_L = 2r$ pour le ton.
   * Applique une limite douce par tangente hyperbolique sans réajustement :
     $$L_{\text{out}} = L + r_L \tanh\!\left(\frac{T(L) - L}{r_L}\right), \qquad \Delta ab_{\text{out}} = \Delta ab \cdot \frac{\tanh(\|\Delta ab\| / r)}{\|\Delta ab\| / r}$$
   * La dérivée de $L_{\text{out}}$ restant comprise entre $1$ et $T'(L) \ge 0$, la courbe reste strictement croissante (zéro repli possible).

### 4.2. Moteur `zones` (`core/palette.py`, `matching.py`, `zone_transfer.py`)
* **Segmentation** : `cv2.kmeans` ($K=8$, graine RNG fixée à $0$) sur la copie Oklab $256\text{ px}$. Fusion des zones $< 1\,\%$ d'aire, tri par $L$ croissant ($1 \dots K$).
* **Appariement de Sinkhorn** : matrice de coût $C_{ij} = 2,0 (\ell_i - \ell_j)^2 + 0,5 \frac{\|ab_i - ab_j\|^2}{s_{ab}^2}$, régularisation entropique $\varepsilon = 0,05$ en domaine logarithmique.
* **Cuisson** : chaque zone applique une carte MKL $2 \times 2$ en chroma vers le mélange de ses cibles ; la luminosité suit par défaut le transfert global monotone (`luminance_strength = None`) pour éviter les replis. Mélange par poids doux (distance de Mahalanobis régularisée par $\sigma_0 = 0,02$, température $\tau$, a priori de surface $\beta = 0,3$, logit global de repli $-4,5$).
* **Conflits** : `find_conflicts` signale les paires de zones dont le coefficient de Bhattacharyya dépasse $0,3$ et dont les déplacements diffèrent de plus de $0,05$. `harmonize_conflicts` fusionne automatiquement les cibles des liens `"auto"` en conflit au chargement.

---

## 5. Pont IA Hors-Ligne & Format `.szl` (v2)

* **Extraction tolérante (`core/recipe.py:extract_json`)** : extrait le **dernier** bloc de code ` ```json ` d'une réponse en texte libre (permettant à l'IA de raisonner avant de conclure).
* **Invariants de `parse_recipe` & `apply_recipe`** :
  * Supporte `RECIPE_MODES = ("zones", "ambiance", "look")`.
  * En mode `"look"`, le bloc `"links"` est ignoré silencieusement même s'il est présent ou mal formé.
  * Si `recipe.analysis_id` diffère de l'empreinte SHA-256 de la paire chargée : en mode `"look"`, tous les réglages (`separation`, `protect_vivid`, `locality`, `global`, `finishing`) sont appliqués **sauf `look.focus`** (les identifiants `R1..Rn` ne désignant plus les mêmes régions), avec notification dans la barre d'état.
  * `merge_finishing` n'écrase que les clés explicitement présentes dans `recipe.finishing_keys`, préservant les roues et la courbe de l'utilisateur lorsque l'IA n'ajuste que `vibrance` ou `saturation`.
* **Garde-fous des Prompts (`core/look_pack.py` et `core/ai_pack.py`)** :
  * Ne jamais conseiller à l'IA de baisser `luminance_strength` ou `chroma_strength` par prudence (les forces se multiplient : $\text{strength} \times \text{chroma\_strength} \times \text{mix}$).
  * L'exemple JSON généré dans `_PROMPT` (`example_recipe`) doit toujours être validé par test unitaire via `parse_recipe`.

---

## 6. Banc à Vérité Connue & Constats Mesurés

Le module `src/stealzalut/devtools/ground_truth.py` évalue les méthodes sur $8$ étalonnages synthétiques connus $G$ en mesurant le **résidu** :
$$\text{résidu} = \frac{\| \text{LUT}_{Q \to G(P)}(Q) - G(Q) \|_{\text{Oklab}}}{\| Q - G(Q) \|_{\text{Oklab}}}$$
($0 =$ étalonnage parfaitement retrouvé ; $1 =$ équivalent à ne rien faire ; $> 1 =$ dégradation).

| Régime (25 photos neutres, LUT $33^3$ Display P3) | `look` (défaut) | `zones` | `oklab` | Constat scientifique |
|---|---|---|---|---|
| **Même contenu ($Q = P$)** | **$0,06\text{ – }0,07$** | $0,13\text{ – }0,17$ | $0,18\text{ – }0,20$ | `look` retrouve quasi parfaitement la courbe de tons et les étirements de chroma, avec $0$ repliement sur toutes les paires. |
| **Autre contenu ($Q \neq P$, localité $0\,\%$)** | $4,1\text{ – }6,3$ | $4,3\text{ – }5,9$ | $4,2\text{ – }5,8$ | Deux photos sans rapport diffèrent $\sim 3\times$ plus entre elles ($0,156$ Oklab) que l'étalonnage ne déplace les couleurs ($0,056$). Tout alignement de distributions copie donc le sujet. |
| **Autre contenu ($Q \neq P$, localité $100\,\%$)** | **$1,45\text{ – }1,60$** | — | — | Le bornage doux (`with_locality`) divise par $3\text{ à }4$ le sur-transfert de sujet, sans coût de calcul ni repliement. |

> **Règle de calibrage** : Ne jamais recalibrer les constantes internes de `core/look.py` pour tenter d'abaisser le score « autre contenu » à localité nulle : c'est une limite fondamentale d'identifiabilité statistique, pas un défaut de constantes. Toute modification algorithmique de `core/` doit maintenir le résidu « même contenu » et respecter les oracles de `tests/test_look.py`.

---

## 7. Pièges Techniques & Invariants du Code

1. **Compteurs de méthodes dans les tests** : `tests/test_color_transfer.py` et `tests/test_gui.py` vérifient `len(SUPPORTED_METHODS) == 11` et `window._algo_combo.count() == 11`.
2. **Constructeur positionnel de `EditState` (`gui/history.py`)** : `EditState` est instancié positionnellement à plusieurs endroits. Tout nouveau champ doit **obligatoirement être ajouté à la fin avec une valeur par défaut**.
3. **Avertissement Qt `QFont::setPointSize <= 0`** : `theme.qss` définit la taille des polices en `px`. Ne jamais utiliser `QPushButton.setMenu()` ni `QFont.setPointSizeF()` sans vérifier `pointSizeF() > 0` ; ouvrir les `QMenu` manuellement via `menu.popup(...)` (comme pour `importRecipeButton`, `focusButton` et `lookRecipeButton`).
4. **Priorité du fichier `.env` local** : `load_settings()` charge `.env` qui prime sur la valeur par défaut du code (`COLOR_TRANSFER_METHOD`). Les tests d'interface qui dépendent d'un algorithme précis doivent le sélectionner explicitement.
5. **Gestion du cycle de vie des `QThread` (`MainWindow._retire_transfer_worker`)** : Un `TransferWorker` remplacé par une nouvelle requête pendant qu'il tourne ne doit jamais être détruit par le ramasse-miettes Python avant sa fin (crash fatal Qt). Ses signaux sont déconnectés et sa référence est conservée dans `_stale_transfer_workers` jusqu'au signal `finished`.