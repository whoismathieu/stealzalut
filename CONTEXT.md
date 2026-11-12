# StealZaLut — contexte condensé

> **À lire en premier, et en général à lire seul.** Ce fichier remplace la lecture des
> ~150 Ko de documents du projet. N'ouvre un document détaillé (§9) que pour la tâche
> précise qui le demande. État au 2026-11-12, branche `approach-v3`, commit `3b6371d`.

## 1. Le produit

Application de bureau **PySide6** (Windows, Python 3.10, `.venv`). Elle prend une photo
**source** et une photo de **référence** et exporte une **LUT 3D `.cube`** qui donne à la
source l'ambiance colorimétrique de la référence. Public : photographes (Lightroom,
Camera Raw, Resolve). Cible de rendu de Mathieu : argentique moyen format de nuit, à la
Greg Girard (couleurs séparées, noirs denses).

Limite physique : un `.cube` est une fonction de la couleur du pixel. Deux régions de
même couleur reçoivent le même traitement. Aucun masque spatial n'y est possible.

## 2. Règles non négociables

- **Aucune mention d'un modèle d'IA comme auteur**, nulle part (commits, PR, fichiers,
  commentaires). Pas de `Co-Authored-By`, pas de « Generated with… ».
- **Jamais `git push`** sans demande explicite de Mathieu. Ne jamais afficher l'URL du
  dépôt distant ni `.git/config` (jeton en clair) : filtrer la sortie de `git push` /
  `git fetch`.
- **Ne pas piocher d'images** ailleurs que dans les dossiers donnés par Mathieu (§7).
  Ne jamais copier ces photos (tierces) ni des planches qui les contiennent dans le dépôt.
- Hors ligne : aucune API d'IA, aucune clé, aucun réseau.
- Interface en **français** ; code, docstrings, commits en **anglais** (`feat:`, `fix:`,
  `docs:`, corps qui dit pourquoi). Un commit par tâche.
- `core/` sans Qt ; `workers/` = `QThread` + `Signal` ; `gui/` = présentation.
- Avant chaque commit : `pytest`, `ruff`, `mypy --strict` verts (565 tests à ce jour).
- Honnêteté du produit : afficher les limites plutôt que les cacher. Ne jamais annoncer
  une qualité sans mesure ou planche.
- Disque presque plein : sorties volumineuses dans un dossier temporaire, hors dépôt.
- Pour écrire un fichier ou un correctif contenant des apostrophes : outil d'écriture,
  pas de *heredoc* bash (a échoué plusieurs fois).

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/python.exe -m mypy src
.venv/Scripts/python.exe -m stealzalut.main
```

## 3. Architecture (`src/stealzalut/`)

Cuisson (`core/grading.py:build_grid(model, size, params, finishing, space=SRGB)`) :
grille neutre → `model` → dosages et protections (`GradingParams`) → lissage et contrôle
de repliement → finition → compression de gamut → `.cube`. Tout le calcul est en
**Oklab**. L'aperçu applique la même grille (sRGB, 33³) à une vignette.

| Module | Rôle |
|---|---|
| `core/color_transfer.py` | `TransferModel` (`apply` sRGB, `apply_lab` Oklab non borné), `fit_transfer`, `SUPPORTED_METHODS` (11) |
| `core/look.py` | Méthode **`look`** (« Rendu », méthode par défaut) : courbe de tons monotone + transport de chroma par bande de tons ; `separation`, `protect_vivid`, `zone_weights`, `look_diagnostics` |
| `core/zone_transfer.py`, `palette.py`, `matching.py`, `correspondence.py` | Méthode `zones` : k-means Oklab, appariement, table de correspondances éditable |
| `core/ambiance.py` | Méthode `ambiance` (références d'une seule teinte) |
| `core/rgb_spaces.py` | `RGBSpace`, `CUBE_SPACES` = sRGB, Display P3, Adobe RGB, Rec.709 γ 2,4 (ProPhoto exclu) |
| `core/colorspace.py` | Oklab, `gamut_compress(lab, iters, to_linear)` |
| `core/finishing.py` | Roues, courbe, 8 bandes, saturation, vibrance |
| `core/recipe.py` | Recettes `.szl` v2 ; modes `zones`, `ambiance`, `look` |
| `core/ai_pack.py`, `look_pack.py` | Packs pour l'IA de l'utilisateur (fichiers + presse-papiers) |
| `core/lut_apply.py` | `apply_lut` (8 bits), `apply_lut_full`, `apply_lut_colors` (flottant) |
| `utils/image_io.py` | `load_image_rgb` → `WideImage` : tableau sRGB 8 bits **pour l'affichage** + `analysis_lab` (couleurs d'origine non écrêtées, 256 px) **pour le moteur** |
| `gui/main_window.py` | Fenêtre (gros fichier : chercher les symboles par nom) ; `gui/history.py` : `EditState` (annulation) |
| `devtools/ground_truth.py` | Banc de mesure à vérité connue (outil, pas utilisé par l'appli) |

Pièges : deux tests comptent les méthodes (11) ; `EditState` est construit
positionnellement (n'ajouter des champs qu'à la fin, avec défaut) ; les menus Qt
s'ouvrent à la main (`QPushButton.setMenu` + feuille de style → avertissement `QFont`) ;
le fichier `.env` local fixe `COLOR_TRANSFER_METHOD` et prime sur le code.

## 4. Ce qui est fait (tout est sur `main` distant sauf mention)

- Audit : le paradigme « zones de couleur appariées » n'est pas le bon moteur par
  défaut ; luminosité par une droite, couleur transférée indépendamment du ton.
- Méthode `look` et son interface : curseur « Séparation », bouton « Référence ▾ »
  (ignorer / privilégier une zone de la référence), bouton « Recette ▾ » (recettes
  `.szl`, pack IA avec le rendu actuel), bandeau des limites.
- Entrée fidèle : profils ICC lus (Display P3, Adobe RGB / ClayRGB / opRGB, ProPhoto,
  par sous-chaîne), couleurs larges analysées sans écrêtage ; bandeau pour les sources
  RAW ; « Exporter l'image… » (rendu pleine résolution, sRGB 8 bits).
- Menu « Espace du .cube » ; le message de fin d'export dit quoi choisir dans l'hôte.
- Banc de mesure en espace large, lots réglage / contrôle.
- **Local seulement, 4 commits non poussés** : banc en espace large, ClayRGB / opRGB,
  mesures de départ, trois essais.

## 5. Constats mesurés (ne pas refaire)

**RAW et Lightroom.** Le `.cube` exporté est identique à l'aperçu au pixel près.
L'écart vu dans Lightroom vient de l'entrée : pour un DNG, l'appli analyse le JPEG
intégré par l'appareil ; Lightroom applique la LUT à son propre développement. D'après
Mathieu (non vérifié ici) : un profil créé dans Camera Raw à partir d'un `.cube` repose
sur **Adobe Standard**, et la LUT passe **après** la courbe des tonalités. Consigne
utilisateur : exporter la photo sous profil neutre, curseurs à zéro, s'en servir comme
source, puis appliquer le profil au RAW.

**Banc à vérité connue** (25 photos neutres ; résidu = erreur / erreur de ne rien
faire ; 0 parfait, > 1 pire que rien) :

| Méthode | Même contenu | Autre contenu |
|---|---|---|
| `look` | 0,11 | 4,1 à 6,3 |
| `zones` | 0,13 à 0,17 | 4,3 à 5,9 |
| `oklab` | 0,21 à 0,24 | 4,2 à 5,8 |

- Même contenu : `look` est le meilleur ; faiblesses : gain de saturation (0,25),
  sarcelle-orange (0,22).
- Autre contenu : les trois méthodes font pire que rien, **à égalité** ; mieux que rien
  dans 13 % des cas seulement. Cause : deux photos sans rapport diffèrent ~3× plus
  (0,156) qu'un étalonnage ne les change (0,056). Rester dans la même catégorie n'aide
  pas. **Ce n'est pas une affaire de constantes : ne pas recalibrer `look` là-dessus.**
- Le banc exagère : copier l'exposition de la référence y compte comme une erreur.

**Trois essais contre ce sur-transfert :**
1. Référence lue contre une « photo moyenne » : **rejeté** (3,4 et même contenu détruit).
2. Ne transférer que points noir / blanc et teinte de l'axe neutre : **rejeté** (1,4 à
   3,4 sur autre contenu ; l'« axe neutre » d'un paysage est son ciel).
3. **Transport local** (une couleur ne regarde que les couleurs de référence à moins
   d'un rayon ρ) : autre contenu ton 1,01 / couleur 1,2–1,3 à ρ = 0,03, mieux que rien
   dans 43–45 % des cas, confirmé sur les deux lots ; mais même contenu dégradé (ton
   0,85, couleur 0,5). ρ = curseur « respecter la photo ↔ coller à la référence ».
   Prototype seulement (`prototypes/bench_local.py`), lent, sans garantie de monotonie.

**Paires réelles source → référence argentique** (25 paires) : les deux méthodes copient
le sujet (ciel bleu → rouge brique, nuages → verts). `zones` colle mieux aux
statistiques mais replie la grille sur 9 paires ; `look` jamais. Le détecteur
« référence monochrome » (`MONOCHROME_CONCENTRATION = 0,85`) se déclenche sur 12 des 25
paires : trop sensible.

## 6. Ouvert

| Sujet | État |
|---|---|
| Intégrer un **rayon de localité** au mode `look`, comme réglage | **reporté par Mathieu** ; à reprendre plus tard |
| Seuil du détecteur monochrome | à revoir |
| Prompt du pack IA du mode `look` | jamais essayé sur une vraie IA |
| Accord du `.cube` avec Lightroom / Resolve (espaces, courbes) | à tester par Mathieu |
| Garder ou retirer la méthode `ambiance` | décision de Mathieu |
| Aperçu et export d'image en espace large | non fait (tout l'affichage est en sRGB 8 bits) |
| Masques à la sortie (plusieurs LUT + caches) | non planifié ; essai grossier peu concluant |

Idée directrice pour la suite : rien d'automatique ne distingue l'étalonnage du sujet à
partir d'une seule image ; ce jugement (« ce vert est de l'herbe ») ne peut venir que de
l'utilisateur ou de son IA.

## 7. Données de test (sur la machine de Mathieu, hors dépôt)

- `C:\Users\mathi\Pictures\test_images\benchmark_photos_source` : 25 photos numériques
  neutres, sans profil ICC (10 nuit, 5 rue, 10 paysage), `manifest.json`.
- `C:\Users\mathi\Pictures\test_images\benchmark_photos_analog` : 30 références
  argentiques (nuit, CineStill, paysages saturés), `manifest.json`.
- Paire historique : `C:\Users\mathi\Pictures\IMG_6295.DNG` (aperçu intégré Display P3).

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m stealzalut.devtools.ground_truth <dossier> \
    [--methods look,zones,oklab] [--lot reglage|controle] [--set BANDS=8] [--json f.json]
```
Compter ~10 minutes par lot ; lancer en arrière-plan.

## 8. Vocabulaire

Oklab (`L` luminosité, `(a, b)` couleur, chroma `C`) · ton = `L` · bande de tons (12
dans `look`) · dominante = déplacement moyen de couleur d'une bande · séparation = poids
du reste du transport · zone = grappe de couleurs k-means · recette = fichier `.szl` ·
pack = dossier remis à l'IA · repliement = luminosité de sortie qui décroît quand
l'entrée croît · résidu = score du banc.

## 9. Documents détaillés (à n'ouvrir qu'au besoin)

| Besoin | Fichier |
|---|---|
| Mesures du banc et des trois essais | `docs/approach-v3/BENCH_BASELINE.md` |
| Ce que fait chaque tâche T01–T11, écarts et réserves | `docs/approach-v3/IMPLEMENTATION_PLAN.md` |
| Signatures, constantes de `look`, textes d'interface mot pour mot | `docs/approach-v3/SPEC.md` |
| Pourquoi la méthode `look` | `docs/approach-v3/DESIGN.md` |
| Diagnostic du moteur d'origine | `docs/audit/AUDIT.md` |
| Oracles sur les cas réels | `docs/approach-v3/ACCEPTANCE.md` |
| Scripts d'essai | `docs/approach-v3/prototypes/` |

`ARCHITECTURE.md`, `README.md` et `stealzalut-advanced-features-architecture.md` sont
**périmés** (application d'origine) : ne pas s'y fier.
