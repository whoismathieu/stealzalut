# Scripts de diagnostic (état au 2026-11-11, commit `fd49d2c`)

Ces scripts ont servi à mesurer, sur de vraies photos, pourquoi les rendus étaient « pas assez intenses » ou « un filtre uniforme ». Ils **tournent encore** contre le code actuel (vérifié : `diag_intensity`, `diag_purple`, `diag_chroma` reproduisent les chiffres du brief). Ils sont des **outils de mesure jetables**, pas du code de production : réutilise-les ou réécris-les mieux.

## Lancer

Chemins **en dur vers la machine de l'utilisateur** (`C:\Users\mathi\Pictures`, `...\Downloads`, `...\Documents\StealZaLut\src`). Chaque script prend **un dossier de sortie** en argument et y écrit ses images / `.npy`. Les scripts qui lisent la recette de l'IA attendent un fichier `ai_answer.txt` dans ce dossier.

```bash
T=$(mktemp -d)                                    # dossier jetable, hors dépôt
cp docs/audit/data/ai_answer_tokyo_green_trimmed.txt "$T/ai_answer.txt"
.venv/Scripts/python.exe -I docs/audit/diagnostics/diag_intensity.py "$T"
```

N'écris pas les sorties dans le dépôt (le disque C: est presque plein : voir le brief).

## Ce que montre chaque script

| Script | Question | Résultat clé |
|---|---|---|
| `diag_tokyo.py` | La recette de l'IA, appliquée telle quelle, change-t-elle quelque chose ? 5 variantes : recette importée / liens IA sans réglages globaux / liens à force 1 / auto / transfert global Oklab. | Dose réelle de couleur 19–54 % selon la zone (les doses se multiplient). |
| `diag_tokyo3.py` | Après correction du bug de force : recette telle quelle vs dosée à 100 %. | Ombres : source +0,004 → recette −0,007 → dosée −0,023 (référence −0,032, sur l'axe a d'Oklab). |
| `diag_intensity.py` | Qu'est-ce qui limite l'intensité ? Un levier à la fois (luminance, transitions τ, dosage). | La chroma moyenne plafonne vers 0,040–0,041 (référence 0,053) quel que soit le levier. |
| `diag_chroma.py` | **Où** la chroma se perd-elle, zone par zone : cible → transfert pur → mélange → gamut → LUT finale. | Perte surtout au passage **gamut sRGB** dans les tons sombres (S1 : 0,037 → 0,027) ; mélange doux peu coupable (poids de sa propre zone 0,74–0,86). |
| `diag_boost.py` | « Intensité couleur » > 100 % ou Finition (vibrance, saturation) comblent-elles l'écart ? | Intensité 130 % : 0,044 ; vibrance +0,5 et saturation +0,15 : 0,049 et 23 % de pixels très colorés (comme la référence). |
| `diag_purple.py` | Référence quasi monochrome (violet) : que devient le transfert par zones ? | Concentration des teintes : référence 0,99, rendu 0,97 : **un filtre uniforme**. 7 zones de référence sur 8 sont violettes ou noires. |
| `diag_vivid.py` | Prototype : protéger les couleurs vives de la source (néons) en mode zones. | Aide un peu (concentration 0,97 → 0,93), insuffisant : blancs, arbres, chaussée restent violets. |
| `diag_ambiance.py` | Prototype « ambiance » (dominante par tons ajoutée aux couleurs de la source) avec 3 teintes fixes par quantiles. | Premier jet, abandonné pour la **courbe de teinte selon la luminosité** (voir `core/ambiance.py`). |
| `check_ambiance.py` | Le mode Ambiance livré (`core/ambiance.py`) rendu à travers la vraie LUT sur les deux cas. | Violet : ambiance franche, néons conservés, mais 99 % de pixels colorés (référence : 80 %). Tokyo vert : trop discret, les Zones restent meilleures. |
| `smoke_gui.py` | Exemple de pilotage de l'interface sans interaction : charge les deux images, vérifie le bandeau « Passer en Ambiance », bascule, capture la fenêtre. Lancement : `PYTHONPATH=src .venv/Scripts/python.exe smoke_gui.py <dossier_jetable>`. | Sert de modèle pour des captures d'écran reproductibles. |

Définitions des métriques utilisées : `L moy` / `C moy` = moyenne de la luminosité / chroma Oklab ; `px C>0.06` = part de pixels très colorés ; « ombres » = pixels de L < 0,35 ; « concentration de teinte R » = longueur résultante moyenne des teintes des pixels colorés pondérée par la chroma (1 = une seule teinte, ~0,35 = palette variée). Version de production : `colour_stats` dans `src/stealzalut/core/palette.py`.
