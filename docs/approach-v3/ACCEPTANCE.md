# Critères d'acceptation v3

## 1. Automatiques (à chaque commit)

```bash
.venv/Scripts/python.exe -m pytest -q          # 393 tests au commit 098396b, puis davantage
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/python.exe -m mypy src
```

## 2. Oracles sur les cas réels

Commande (les photos sont dans le dossier Images de Mathieu, chemins dans
`prototypes/harness.py`) :
```bash
PYTHONPATH=src .venv/Scripts/python.exe docs/approach-v3/prototypes/run_p1.py <dossier_jetable>
```
`run_p1.py` utilise le prototype (`look_proto.py`) ; les valeurs ci-dessous sont celles
du **moteur livré** (`fit_look` de `core/look.py`), rendu à travers la LUT 33³ sur la
vignette de 640 px. Elles ont été calculées, pas estimées. Toute modification du moteur
doit rester dans les tolérances, ou justifier l'écart par une nouvelle planche.

### Cas A — `IMG_6295.DNG` × `ben-blennerhassett-CptmwIhxfo4-unsplash.jpg`

| Mesure | Référence | Attendu (`look`, défauts) | Tolérance |
|---|---|---|---|
| L moyenne | 0,280 | 0,296 | ± 0,015 |
| 95ᵉ centile de L | 0,64 | 0,86 | ≥ 0,80 |
| Chroma moyenne | 0,053 | 0,045 | ≥ 0,043 |
| Pixels colorés (C ≥ 0,05) | 43,4 % | 28,2 % | ≥ 26 % |
| Ombres (L < 0,35), axe a | −0,032 | −0,031 | ± 0,004 |
| Ombres, axe b | +0,015 | +0,015 | ± 0,004 |
| Repliements (`fold_fraction`) | — | 0 | = 0 |
| `protect_vivid` du modèle | — | 0,0 | = |
| `model.tone([0,2 ; 0,4 ; 0,6 ; 0,8 ; 1,0])` | — | 0,105 ; 0,225 ; 0,444 ; 0,724 ; 0,887 | ± 0,01 |
| `model.cast[0]`, `cast[5]`, `cast[11]` | — | (−0,031 ; 0,031), (−0,042 ; 0,023), (−0,008 ; −0,007) | ± 0,004 |
| Séparation 0 → pixels colorés | — | 21,6 % | ± 2 points |
| Séparation 1,5 → pixels colorés | — | 31,6 % | ± 2 points |

### Cas B — `IMG_6295.DNG` × `841121141_…_n.jpg`

| Mesure | Référence | Attendu | Tolérance |
|---|---|---|---|
| L moyenne | 0,273 | 0,301 | ± 0,015 |
| 95ᵉ centile de L | 0,42 | 0,84 | ≥ 0,75 |
| Chroma moyenne | 0,117 | 0,105 | ≥ 0,100 |
| Pixels colorés | 80,4 % | 88,5 % | ≤ 92 % |
| Concentration de teinte | 0,99 | 0,93 | 0,90–0,96 |
| Ombres, axes a / b | +0,052 / −0,091 | +0,052 / −0,088 | ± 0,005 |
| Repliements | — | 0 | = 0 |
| `protect_vivid` du modèle | — | 0,8 | = |

### Performances (machine de Mathieu)
Ajustement ≤ 0,5 s (mesuré 0,30–0,36) ; cuisson 33³ ≤ 0,3 s (0,21) ; 64³ ≤ 0,8 s (0,5).

## 3. Validation visuelle par Mathieu (obligatoire)

Les métriques ne suffisent pas quand le contenu des deux photos diffère. À regarder :

| Planche | Question |
|---|---|
| `prototypes/out/p1_case_A.jpg` | Le rendu (en bas à droite) a-t-il l'ambiance de la référence ? Les enseignes blanches sont-elles acceptables (légèrement rosées) ? |
| `prototypes/out/p1_case_B.jpg` | Est-ce encore « un filtre » ? Les blancs sont-ils assez neutres ? |
| `prototypes/out/p2_pairs_1-3.jpg`, `p2_pairs_4-6.jpg` | Sur chaque paire : préfères-tu « ACTUEL zones » ou « LOOK » ? La paire du bas de la seconde planche (photo de jour) est le cas le plus faible. |
| `prototypes/out/p3_masks.jpg` | Le composite avec masque (à droite) vaut-il l'effort par rapport à la LUT seule ? |

Puis dans l'application : méthode « Rendu (tons, dominantes et séparation de la
référence) », les deux photos, et l'export `.cube` ouvert dans Lightroom ou Resolve.

Réponses attendues de Mathieu, qui débloquent le plan :
1. `look` devient-il la méthode par défaut ? **Oui** (2026-11-11, T08 fait).
2. Le mode `ambiance` peut-il disparaître de la liste ?
3. La phase 2 (masques à la sortie) vaut-elle d'être tentée ?

## 4. Par tâche

| Tâche | Accepté quand |
|---|---|
| T03 | Curseur visible seulement en mode Rendu ; le déplacer change l'aperçu en moins de 0,5 s ; Ctrl+Z le ramène. |
| T04 | Paire n° 6 : bandeau « La photo a peu de couleurs… » ; cas A : pas de bandeau. |
| T05 | Cas B : ignorer la zone du cadre noir de la référence ne change pas le rendu de plus de 0,01 en L moyenne (elle pèse peu), ignorer la zone du ciel le change visiblement (planche à joindre). |
| T06 | Une recette exportée en mode Rendu se réimporte à l'identique ; les recettes de `docs/audit/data/` s'importent comme avant. |
| T07 | Une vraie réponse d'IA s'importe sans erreur sur A et B ; planche avant / après jointe. |
