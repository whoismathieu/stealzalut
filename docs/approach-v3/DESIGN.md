# Approche v3 — « Rendu » : lire la référence comme un tirage, pas comme une palette

> Suite de `docs/audit/AUDIT.md`. Tout ce qui est affirmé ici a été exécuté :
> scripts dans `prototypes/`, planches dans `prototypes/out/`, moteur dans
> `src/stealzalut/core/look.py` (commit `098396b`).

## 1. L'idée en une page

Un `.cube` ne voit que la couleur de chaque pixel. La question n'est donc pas « quelle
zone va vers quelle zone » mais : **que peut-on apprendre d'une référence qui s'exprime
comme une fonction de la couleur, et qu'un tireur reconnaîtrait ?** Quatre choses, dans
l'ordre où un coloriste les règle :

| # | Ce que l'on apprend | Dans un tirage argentique de nuit | Dans le moteur |
|---|---|---|---|
| 1 | **Courbe de tons** | noirs denses, pied de courbe, lumières qui restent lumineuses | courbe monotone `L → L'` |
| 2 | **Dominante selon le ton** | ombres vertes (tubes fluorescents), hautes lumières chaudes ou mauves : les couches ne se croisent pas au même endroit | déplacement moyen de `(a, b)` par bande de tons |
| 3 | **Quantité de couleur selon le ton** | ombres encore colorées, tons moyens saturés | portée par le transport (point 4) |
| 4 | **Séparation** | éclairages mixtes écartés les uns des autres au lieu d'être fondus par une balance des blancs | transport de la *distribution* de chroma, bande par bande |

Le point clé, celui qui lève le plafond mesuré dans l'audit : **la couleur est apprise
conditionnellement au ton de sortie.** Pour chaque bande de tons de la source, on regarde
quelles couleurs la référence montre *au ton où cette bande est envoyée*, et on y
transporte la distribution de chroma de la source. La couleur visée existe donc, par
construction, à la luminosité obtenue : le gamut n'a presque plus rien à raboter.

La « ségrégation des couleurs » demandée s'obtient **dans l'espace des couleurs** : une
LUT ne peut séparer que ce qui diffère déjà par (L, a, b), mais elle peut *amplifier* ces
différences. Une lumière un peu chaude et une lumière un peu froide, presque grises
toutes les deux dans la photo numérique, sont envoyées vers deux pôles distincts de la
palette de la référence. C'est exactement ce que fait une pellicule lumière du jour sous
un éclairage mixte. Le réglage qui dose cet écartement s'appelle **Séparation**.

## 2. L'algorithme

Notations : Oklab `(L, a, b)` ; copies d'analyse de 256 px (`ANALYSIS_MAX_SIZE`).

### 2.1. Ton
- Quantiles de `L` de la source et de la référence aux niveaux 1, 5, 15, 30, 50, 70,
  85 % (**la masse** de l'image) et 99,9 % (**le point blanc**).
- Nœuds `(0, 0)`, les 8 paires, puis `x = 1`. Chaque nœud est borné pour que la pente
  depuis le précédent reste dans [0,35 ; 3].
- Courbe cubique monotone (Fritsch–Carlson, déjà dans `core/finishing.py:eval_curve`).

Pourquoi sauter de 85 % à 99,9 % : entre les deux se trouvent les **sources lumineuses**
(enseignes, fenêtres). Leur *surface* dépend du sujet, pas du rendu. Une égalisation
complète des histogrammes rendait gris-rose les enseignes blanches du cas A (la
référence a peu de lumières) ; mesuré : 95ᵉ centile de L à 0,65 au lieu de 0,86.

### 2.2. Couleur par bande de tons
12 bandes centrées sur les quantiles `(m + 0,5)/12` de `L` source. Pour la bande `m`,
de centre `ℓ_m`, envoyée au ton `ℓ'_m = T(ℓ_m)` :

1. poids gaussiens en `L` autour de `ℓ_m` (source) et de `ℓ'_m` (référence), largeur =
   écart aux bandes voisines, au moins 0,02 ;
2. tirage systématique de 3000 points `(a, b)` de chaque côté ;
3. **transport tranché** : 48 directions (suite du nombre d'or), le long de chacune les
   projections sont appariées rang par rang. Résultat : pour chaque point source, une
   cible ; l'ensemble des cibles a la distribution de la référence ;
4. **dominante** de la bande = déplacement moyen ; **résidu** = le reste ;
5. le résidu est déposé sur un treillis 49 × 49 en chroma **compandée**
   `p = ab / (|ab| + 0,05)` (mailles fines près du gris, où vivent les pixels d'une
   photo), lissé (σ = 1,5 maille) ; les couleurs absentes de la source reçoivent un
   champ lointain (σ = 6 mailles) qui tend vers zéro ;
6. **confiance** : si la référence a moins de 0,4 % de ses pixels à ce ton, le
   déplacement de la bande est réduit (carré du rapport) : la référence « ne dit rien ».

Dominantes et résidus sont lissés entre bandes voisines (σ = 0,7 bande).

### 2.3. Application à une couleur
```
L'      = T(L)
(a, b)' = (a, b) + [ dominante(L) + séparation × résidu(L, a, b) ] × (1 − protection × vive(C))
```
puis compression de gamut à teinte constante (existante). `vive(C)` = `smoothstep(C ;
0,04 ; 0,12)`. **Protection des couleurs vives** = 0,8 si la référence n'a qu'une teinte
(`ColourStats.is_monochrome`), sinon 0 : c'est ce qui évite le « filtre » du cas B sans
mode séparé.

### 2.4. Propriétés garanties
- **Aucun repliement possible** : `L'` ne dépend que de `L`, par une courbe monotone
  (mesuré : 0 sur les huit paires, en 17³ et 33³).
- **Déterministe** : pas de tirage aléatoire.
- **Identité** : source = référence donne un écart moyen < 0,01 (test).
- C'est un `TransferModel` ordinaire : dosage, protections, lissage, finition, aperçu et
  export fonctionnent sans modification.

## 3. Preuves

Mesures à travers la vraie LUT 33³, moteur livré (`core/look.py`), réglages par défaut.

| Cas A — Tokyo vert | L moy. | 95ᵉ c. de L | Chroma | Px colorés | Ombres (a) | Repl. |
|---|---|---|---|---|---|---|
| Référence | 0,280 | 0,64 | 0,053 | 43,4 % | −0,032 | — |
| Actuel (zones + recette IA à 100 %) | 0,292 | 0,89 | 0,041 | 19,0 % | −0,022 | 0,0043 |
| **Rendu** | 0,296 | 0,86 | 0,045 | 28,2 % | **−0,031** | **0** |
| Rendu, séparation 0 | 0,296 | 0,86 | 0,044 | 21,6 % | −0,029 | 0 |
| Rendu, séparation 1,5 | 0,296 | 0,86 | 0,046 | 31,6 % | −0,031 | 0 |

| Cas B — nuit violette | L moy. | 95ᵉ c. de L | Chroma | Px colorés | Concentration |
|---|---|---|---|---|---|
| Référence | 0,273 | 0,42 | 0,117 | 80,4 % | 0,99 |
| Actuel (ambiance) | 0,273 | 0,53 | 0,110 | 99,4 % | 0,94 |
| **Rendu** | 0,301 | 0,84 | 0,105 | **88,5 %** | 0,93 |

Planches : `out/p1_case_A.jpg`, `out/p1_case_B.jpg` (cas du brief), `out/p2_pairs_*.jpg`
(six autres paires), `out/p3_masks.jpg` (masques). Durées : ajustement 0,30–0,36 s,
cuisson 33³ 0,21 s, 64³ 0,5 s.

**Lecture honnête.** Sur A et B le rendu est nettement plus proche de la référence
(ombres, densité, blancs). Sur les six autres paires : plus propre ou plus proche sur
quatre, équivalent sur une, **plus terne sur une** (source de jour presque grise). Le
nombre de pixels *très* colorés de la référence n'est pas atteint (cas A : 13,9 % contre
23 %) : la source n'a pas la matière. Ces jugements portent sur des planches réduites ;
ils sont à confirmer par Mathieu.

## 4. Où la chroma se perd encore (cas A, `diag_loss.py`)

| Étape | Chroma moyenne |
|---|---|
| Référence | 0,0531 |
| Champ ajusté, avant gamut | 0,0493 |
| Après compression de gamut | 0,0469 |
| LUT 64³ | 0,0464 |
| LUT 33³ | 0,0454 |

L'essentiel de l'écart restant (0,0531 → 0,0493) vient du lissage du champ : c'est le
prix de la régularité (pas d'amplification du bruit). La finition (vibrance) reste le
bon outil pour le combler au goût de l'utilisateur.

## 5. Alternatives écartées

1. **Réparer les zones** (courbe de tons + chroma conditionnelle dans `zone_delta`).
   Rejetée comme moteur par défaut : le cas monochrome reste un filtre, k-means reste
   instable, et l'on garde trois réglages qui se multiplient. Les zones restent
   disponibles pour qui veut choisir les correspondances à la main.
2. **Transport optimal 3D direct** (toute la couleur d'un coup, à la Pitié). Rejetée :
   la luminosité dépendrait de la couleur → repliements (piège n° 1 du brief), et rien
   n'est interprétable ni dosable. Le découpage ton puis chroma donne les mêmes
   statistiques avec une garantie de monotonie et deux réglages lisibles.

## 6. Masques : ce qui est vrai, ce qui ne l'est pas

- **Dans un `.cube`, un masque spatial est impossible.** Aucune astuce ne change cela.
- **Ajuster une LUT sur des cibles masquées (H2) dégrade le résultat** : là où deux
  régions de même couleur veulent deux traitements, la LUT fait la moyenne, donc
  désature (mesuré : pixels colorés 28 % → 16 %).
- **Ce que les masques apportent vraiment au `.cube` : choisir ce que l'on regarde.**
  `fit_look(..., reference_weights=…)` accepte un poids par pixel de la référence. Les
  zones existantes fournissent déjà des masques (`ZonePalette.labels`) : « ignore le
  cadre noir », « lis surtout le ciel ». C'est la tâche T05 du plan.
- **Masques à la sortie (plusieurs LUT + caches, ou image gradée)** : seul moyen de
  traiter différemment deux régions de même couleur. L'essai avec un masque haut/bas
  montre un écart mesurable (0,045–0,070 RMS) mais peu de gain visible ; un essai
  concluant demande des masques sémantiques (segmentation apprise ou pinceau). **Non
  retenu dans cette version** ; décrit comme phase 2 dans le plan, sans engagement.

## 7. Le pont IA (conçu, non validé sur une vraie IA)

Le moteur n'a plus besoin de l'IA pour atteindre la référence ; l'IA devient ce qu'elle
sait faire : **juger un rendu et le corriger avec des mots de coloriste**.

- Le pack contient le **rendu actuel** (`7_current_render.jpg`) et un tableau d'écarts
  mesurés (ton, chroma, dominante des ombres / tons moyens / hautes lumières).
- La recette gagne un mode `"look"` : séparation, protection des vives, zones de la
  référence à ignorer ou à privilégier, réglages globaux, finition. Aucune valeur de
  couleur brute n'est demandée ; les roues de finition existantes (teinte en degrés,
  quantité) suffisent pour « ombres plus vertes ».
- Boucle : rendu → pack → réponse → rendu. Deux ou trois tours.

MCP local : écarté pour l'instant (ajoute une dépendance et un serveur pour un gain non
démontré). À rouvrir si Mathieu utilise un client compatible.

## 8. Ce qui reste impossible, et comment l'application le dit

| Limite | Message (tâche T04) |
|---|---|
| Source presque grise : peu à séparer | « La photo a peu de couleurs : la séparation aura peu d'effet. » |
| La référence n'a presque rien à certains tons | « La référence ne montre presque rien dans les tons … : ces tons gardent leurs couleurs. » |
| Deux régions de même couleur | inchangé : c'est la limite du `.cube`, à rappeler dans l'aide du mode |
| Grain, halation | hors LUT ; non proposé |

## 9. Non testé

Peau et portraits ; paysages de jour colorés ; densité soustractive (couleurs saturées
plus sombres) ; comportement de la LUT sur d'autres photos d'une même série ; vraie
réponse d'IA au nouveau prompt.
