# StealZaLut

> **Générateur de LUTs 3D (`.cube`) par transport optimal conditionnel et analyse perceptuelle en espace Oklab non borné.**
> Application de bureau (Python 3.10+ / PySide6) conçue pour transférer le rendu colorimétrique d'une photographie de référence (notamment le tirage argentique moyen format de nuit) vers une photographie source, tout en respectant la physique des espaces de couleur et les pipelines des logiciels hôtes (Lightroom Classic, Adobe Camera Raw, DaVinci Resolve).

---

## 1. Philosophie & Physique d'une LUT 3D

Une table de correspondance 3D (`.cube`) est une fonction purement colorimétrique $f : [0, 1]^3 \to [0, 1]^3$ évaluée pixel par pixel par interpolation trilinéaire, **sans aucune information spatiale** : deux régions de même couleur d'entrée reçoivent nécessairement la même couleur de sortie.

Pour éviter l'effet « filtre monochrome » ou l'écrasement des hautes lumières typiques des transferts statistiques globaux, **StealZaLut** sépare l'analyse de l'image de sa restitution d'affichage :

1. **Analyse en gamut natif non écrêté (`WideImage`)** : les profils ICC intégrés (Display P3, Adobe RGB, ProPhoto) sont décodés directement vers l'espace perceptuel **Oklab** en virgule flottante (`float64`), préservant les couleurs vives (néons, enseignes) qu'une conversion sRGB 8 bits précoce détruirait.
2. **Découplage Ton / Chroma conditionnel (`look`)** : la luminosité $L$ suit une courbe cubique strictement monotone (aucune inversion de polarité tonale possible), tandis que la chrominance $(a, b)$ est transportée bande de tons par bande de tons vers la luminosité de sortie effective.
3. **Chaîne de cuisson unifiée (`build_grid`)** : le modèle, les dosages, la protection des tons chair/neutres/vifs, le contrôle de repliement (*folding*), et la finition sont évalués en Oklab non borné. La compression de gamut à teinte constante n'est appliquée qu'à la toute dernière étape vers l'espace cible du `.cube`.

---

## 2. Fonctionnalités Clés

### Moteurs de Transfert Colorimétrique (11 méthodes)
* **Rendu (`look` — méthode par défaut)** : lit la référence comme un tirage photographique en 4 composantes :
  * *Courbe de tons monotone* (interpolation de Fritsch–Carlson sur les quantiles $1\,\%$ à $85\,\%$ pour la masse de l'image, puis saut au point blanc à $99,9\,\%$ pour préserver l'éclat des sources lumineuses indépendamment de leur surface).
  * *Dominante par bande de tons* (déplacement moyen de $(a, b)$ sur 12 bandes de luminosité).
  * *Étirement linéaire $2 \times 2$ (`stretch`)* et *champ résiduel sur treillis compandé* ($49 \times 49$ avec $p = \frac{ab}{\|ab\| + 0,05}$, offrant une résolution maximale près de l'axe neutre) calculés par transport optimal tranché (*Sliced Optimal Transport*, 48 projections).
  * *Contrôles dédiés* : **Séparation** ($\in [0, 1,5]$), **Fidélité à ma photo / Localité** ($\in [0, 1]$, bornage doux par $\tanh$ empêchant le sur-transfert de sujet quand source et référence montrent des scènes sans rapport), et **Ciblage par zone de référence** (`Référence ▾` : ignorer un cadre ou privilégier un ciel).
* **Zones (`zones`)** : segmentation déterministe par *k-means* en Oklab ($K \in [2, 16]$, défaut $8$), appariement automatique par transport optimal entropique de Sinkhorn, et éditeur interactif de **Table de Correspondances** (liens pondérés, verrouillage par zone, intentions qualitatives, détection des conflits de Bhattacharyya).
* **Ambiance (`ambiance`)** : conçu pour les références quasi monochromes (concentration de teinte $\ge 0,85$). Extrait une courbe de teinte continue selon $L$ et l'ajoute aux couleurs sources tout en protégeant les couleurs vives à $80\,\%$.
* **Oklab Découplé (`oklab`)** : transport optimal linéaire gaussien séparant l'échelle de luminosité $L$ et la matrice $2 \times 2$ de Monge–Kantorovitch sur $(a, b)$.
* **Méthodes Globales Classiques (`color-matcher`)** : `mkl`, `reinhard`, `hm`, `mvgd`, `hm-mkl-hm`, `hm-mvgd-hm`, `default`.

### Espaces d'Export `.cube` & Aperçu WYSIWYG
* **Résolutions de grille** : $17^3$ ($4\,913$ points), $33^3$ ($35\,937$ points), ou $64^3$ ($262\,144$ points).
* **Espaces colorimétriques du `.cube`** : `sRGB`, `Display P3`, `Adobe RGB (1998)`, `Rec.709 (gamma 2.4)`. *(ProPhoto RGB est exclu de l'export `.cube` car ses primaires imaginaires génèrent des réponses cônes LMS négatives aux sommets du cube $[0, 1]^3$, provoquant un repliement de luminosité dès la grille neutre).*
* **Aperçu fidèle au pixel près** : l'aperçu applique la grille 3D calculée dans l'espace choisi directement aux pixels natifs de l'image source par interpolation trilinéaire avant conversion sRGB pour l'écran.
* **Export d'image pleine résolution** : export direct de la photo source étalonnée en JPEG (qualité 95), TIFF ou PNG.

### Outils de Réglage & Finition (Oklab / Oklch)
* **Ajustements globaux** : Intensité luminance, Intensité couleur, Préserver les neutres, Protection peau (fenêtre Oklch calibrée sur $h \approx 37,5^\circ\text{–}77,5^\circ$ sans bloquer $L$), Préserver les couleurs vives, Lissage 3D et Mix.
* **Panneau Finition** : 3 roues colorimétriques (Ombres / Tons moyens / Hautes lumières pondérées par polynômes de Bernstein sur $L$), courbe tonale cubique monotone sur $L$, 8 bandes de teinte TSL gaussiennes, Saturation et Vibrance.

### Pont IA Hors-Ligne & Recettes `.szl` (v2)
* **Zéro API, zéro réseau** : l'application génère un dossier d'analyse (`_stealzalut_look_pack` ou `_stealzalut_pack` + archive `.zip`) contenant les photos, le **rendu actuel** (`3_current_render.jpg` en mode Rendu), les planches de zones, les mesures d'écarts (`analysis.json`) et le prompt pré-rempli copié dans le presse-papiers.
* **Import intelligent** : collez directement la réponse complète de votre IA (Claude, ChatGPT, Gemini) ; StealZaLut extrait et valide le dernier bloc JSON `.szl`, fusionne la finition et détecte les dosages trop prudents (« Doser à $100\,\%$ »).

---

## 3. Intégration Lightroom Classic / Camera Raw & Fichiers RAW

Lorsqu'un fichier RAW/DNG est chargé directement dans StealZaLut, l'application extrait l'aperçu JPEG intégré par le boîtier (souvent en Display P3 avec une courbe tonale constructeur déjà appliquée). Or, dans Adobe Camera Raw / Lightroom :
1. Un profil `.xmp` créé à partir d'un `.cube` repose sur le socle **Adobe Standard** (et non *Adobe Couleur* ou *Apple ProRAW*).
2. La LUT 3D est évaluée **après** le profil de base et la courbe des tonalités, dans l'espace colorimétrique sélectionné lors de la création du profil.

**Protocole pour une correspondance exacte entre l'aperçu StealZaLut et Lightroom :**
1. Dans Lightroom / Camera Raw, appliquez le profil **Adobe Standard** à votre RAW avec tous les curseurs à zéro.
2. Exportez une copie intermédiaire (JPEG ou TIFF, idéalement en **Display P3**).
3. Utilisez cet export comme **Image Source** dans StealZaLut et exportez votre `.cube` en choisissant l'espace **Display P3**.
4. Dans Camera Raw, créez votre profil en sélectionnant l'espace **Display P3** pour la table de couleurs, puis appliquez ce profil directement à vos fichiers RAW.

---

## 4. Installation & Démarrage Rapide

### Prérequis
* **Python 3.10+** et le gestionnaire de paquets [`uv`](https://docs.astral.sh/uv/).

### Installation (Windows PowerShell)
```powershell
uv sync --all-extras
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

### Lancement de l'application
```powershell
# Via l'environnement virtuel local
.venv\Scripts\python.exe -m stealzalut.main

# Ou via uv / script de lancement
uv run stealzalut
.\launch.bat
```

---

## 5. Tests, Qualité de Code & Banc de Mesure

Toute modification du code doit maintenir au vert les trois vérifications suivantes :

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/python.exe -m mypy src
```

Pour évaluer objectivement les algorithmes de transfert sur un dossier d'images via le **banc de mesure à vérité connue** (en Display P3 flottant, lots `reglage` et `controle`) :

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m stealzalut.devtools.ground_truth <dossier_images> --methods look,zones,oklab
```

---

## 6. Documentation Technique

Toute l'architecture interne, les équations mathématiques, la gestion des espaces de couleur et les invariants de développement sont détaillés dans **[ARCHITECTURE.md](ARCHITECTURE.md)**.