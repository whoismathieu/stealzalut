# Audit de StealZaLut — état au commit `fd49d2c`

> Périmètre : moteur de transfert (`src/stealzalut/core/`), pont IA, finition.
> Mesures : scripts rejouables de `docs/audit/diagnostics/` (existants) et de
> `docs/approach-v3/prototypes/` (nouveaux). Planches : `docs/approach-v3/prototypes/out/`.
> Les chiffres notés « brief » n'ont pas été remesurés ; tous les autres l'ont été le 2026-11-11.

## 1. Verdict

1. **Le paradigme « grappes de couleurs → zones appariées » n'est pas le bon moteur par
   défaut pour l'objectif visé.** Il copie une *palette* zone par zone. Or ce que l'on
   reconnaît dans une référence (et dans un tirage argentique de nuit) est un *rendu* :
   une courbe de tons, une dominante qui change avec le ton, une quantité de couleur qui
   change avec le ton, et un écartement des teintes. Aucune de ces quatre choses n'est
   une « zone ».
2. **Trois défauts de construction, indépendants de l'IA et du prompt, expliquent les
   constats de Mathieu** (« pas assez intense », « filtre », « edge cases ») : la
   luminosité est transférée par une simple droite ; la couleur est transférée
   indépendamment de la luminosité (d'où des cibles hors gamut qu'on rabote ensuite) ;
   la dominante d'une référence est estimée par des moyennes qui mélangent tout.
3. **Une LUT seule peut faire nettement mieux** que le rendu actuel, sans masque : c'est
   prouvé par prototype sur les deux cas du brief et vérifié sur six autres paires (§5).
   Les masques spatiaux ne sont pas la cause du plafond actuel ; leur apport réel reste
   à démontrer avec de vrais masques sémantiques (§4, H2/H3).

Le moteur de zones n'est pas à jeter : il reste le seul qui laisse choisir « ceci va
vers cela », et sur une paire (n° 4, §5) il donne un rendu plus franc. Il devient une
méthode parmi d'autres, plus la méthode par défaut.

## 2. Ce qui marche (à conserver)

| Élément | Constat |
|---|---|
| Aperçu = grille LUT réelle, trilinéaire | Fidèle à l'export. Rien à changer. |
| Oklab, champ de déplacement, compression de gamut à teinte constante | Bonne base ; réutilisée telle quelle par la v3. |
| Chaîne `build_grid` (dosage → lissage → finition → gamut) | Tout `TransferModel` en profite ; la v3 s'y branche sans la modifier. |
| Finition (roues, courbe, bandes, saturation, vibrance) | Fonctionne, sert de « dernier kilomètre ». |
| Choix des correspondances par l'IA (cas A) | Bon (brief §5). Le pont IA n'est pas le maillon faible. |
| Mesure des conflits et des repliements, affichées | Principe produit à garder. |

## 3. Ce qui plafonne, et pourquoi (causes racines)

### 3.1. Physique de la LUT — limite dure

Une LUT est une fonction de la couleur du pixel. Deux pixels de même couleur reçoivent
le même traitement, où qu'ils soient. Dans la source de test, le ciel noir, les façades
sombres et la chaussée ont la même couleur : aucune méthode ne peut donner un ciel mauve
et des façades vertes avec un seul `.cube`. Cela ne se contourne pas, cela s'affiche.

Conséquence moins évidente : **la seule information dont dispose une LUT pour séparer
deux choses est (L, a, b)**. Le moteur actuel n'exploite presque que L (les 8 zones d'une
image peu colorée sont surtout des tranches de luminosité) ; les petits écarts de teinte
autour du gris (lumière chaude contre lumière froide) ne sont pas amplifiés. C'est
pourtant ce que fait une pellicule sous éclairage mixte.

### 3.2. Paradigme des zones

- **Luminosité = une droite.** `OklabTransferModel` applique `L' = μr + κ(L − μs)`
  (`core/color_transfer.py`, classe `OklabTransferModel`). Une droite ne peut pas donner
  à la fois des noirs denses et des lumières qui restent lumineuses. Cas A : la source
  (aperçu iPhone, ombres relevées) a 5 % de pixels au-dessus de L = 0,93, la référence
  5 % au-dessus de 0,64 seulement. La droite assombrit tout ou rien.
- **Couleur indépendante du ton.** Chaque zone envoie sa chroma vers celle de sa cible,
  mais la luminosité du pixel suit un autre chemin (transfert global). La couleur visée
  n'existe donc pas forcément à la luminosité obtenue : `gamut_compress` la rabote.
  C'est la vraie cause du « plafond à 75–78 % » mesuré dans le brief (zone S1 :
  0,037 → 0,027) : ce n'est pas le gamut sRGB en soi (les pixels de la référence sont
  dans le gamut), c'est l'**incohérence entre L et la chroma visée**. Preuve : en
  transférant la chroma *conditionnellement au ton de sortie*, la perte au gamut tombe à
  0,0024 de chroma moyenne (cas A, `diag_loss.py` : 0,0493 → 0,0469), et la teinte des
  ombres atteint −0,031 (référence −0,032, actuel −0,022).
- **Copier une palette, c'est faire un filtre quand la palette n'a qu'une couleur**
  (cas B, brief §5). Confirmé ; défaut de paradigme.
- **k-means sur 256 px découpe le bruit** (piège 4 du brief) ; `harmonize_conflicts`
  est un pansement. Sur la paire n° 1 (§5) les zones automatiques produisent des plaques
  orange et bleu-vert dans le ciel.
- **Le lissage et la force par zone** interagissent mal (pièges 1–3) : la séparation
  obtenue dépend d'un réglage (`tau`) que ni l'utilisateur ni l'IA ne savent régler.

### 3.3. Mode ambiance

Bonne intuition (dominante selon le ton), mauvais estimateurs :
- luminosité : la même droite que ci-dessus ;
- dominante : moyenne des `(a, b)` par ton avec un a priori de 25 pseudo-pixels vers la
  moyenne globale → les tons rares (blancs, noirs profonds) héritent de la couleur
  dominante. D'où les blancs violets et les **99 % de pixels colorés** (référence 80 %) ;
- la quantité de couleur n'est pas transférée (la source est atténuée de 35 %, point).
Sur une référence variée, la moyenne par ton mélange des teintes opposées et s'annule :
c'est pourquoi il est « plus faible que zones » sur le cas A.

### 3.4. Chaîne d'image

- **Tout est en sRGB 8 bits**, les DNG passent par leur aperçu JPEG déjà « développé »
  (ombres relevées, HDR). Conséquence mesurée : la source de test a un ton très différent
  d'un négatif (L de 0,20 à 0,93 entre les 5ᵉ et 95ᵉ centiles) ; il faut une vraie courbe
  de tons pour retrouver des noirs. Pas bloquant, mais c'est une raison de plus de
  traiter le ton en premier.
- **Grille 33³** : coûte 0,0015 de chroma moyenne sur le cas A par rapport au modèle
  appliqué directement (0,0469 → 0,0454 ; en 64³ : 0,0464). Marginal.

### 3.5. Pont IA

- L'IA travaille **à l'aveugle, en un coup** : elle ne voit jamais le rendu.
- Le format (identifiants de zones + niveaux qualitatifs) lui interdit d'exprimer ce
  qu'elle voit très bien : « ombres vert bouteille, hautes lumières mauves, garder les
  blancs ». La règle « jamais de valeurs de couleur » venait de la v1 (lift/gamma mal
  estimés) ; elle ne vaut pas pour un vocabulaire de coloriste (teinte nommée par
  tranche de tons, quantité en pourcentage).
- Elle hérite du plafond du moteur : même avec les bonnes correspondances, cas A plafonne.

### 3.6. Dosage et finition

- Les doses se multiplient (brief) : corrigé, mais le correctif (bouton « Doser à
  100 % ») traite le symptôme. Un moteur dont le réglage par défaut vise déjà les
  statistiques de la référence n'a pas besoin de ce bouton.
- « Vibrance +0,5 / saturation +0,15 » rattrapent le chiffre (23 % de pixels très
  colorés) en saturant *tout* uniformément : ce n'est pas la référence.

## 4. Registre des hypothèses du brief

| | Hypothèse | Statut | Preuve |
|---|---|---|---|
| H1 | Le paradigme des zones n'est pas le bon | **Confirmée** | §3.2 ; prototype `look` meilleur sur A et B sans aucune zone (§5). |
| H2 | Ajuster la LUT sur des cibles construites avec des masques | **Réfutée en l'état** | `run_p3.py` : la régression moyenne les cibles en conflit et **désature** (cas A : chroma 0,045 → 0,040, pixels colorés 28 % → 16 %). Elle n'a d'intérêt que pour *pondérer* ce qu'on regarde (voir DESIGN §6). |
| H3 | Sorties au-delà du `.cube` (masques, plusieurs LUT) | **Non conclue** | Avec un masque grossier haut/bas, le composite diffère du `.cube` seul de 0,045 à 0,070 (RMS Oklab) mais le gain visuel est faible (`out/p3_masks.jpg`). Un vrai test demande des masques sémantiques. Reportée en phase 2. |
| H4 | Traits argentiques dans une LUT | **Partiellement confirmée** | Courbe de tons, dominante par ton, saturation par ton, écartement des teintes : tous réalisables et appris de la référence (prototype). Densité soustractive : non testée. Grain, halation : hors LUT, par nature. |
| H5 | Segmentation apprise | **Non testée** | Hors budget ; n'a de sens qu'avec H3. |
| H6 | Lire la référence comme un style | **Confirmée** | C'est l'approche retenue (transport par tons). |
| H7 | Donner à l'IA de quoi voir son résultat | **Non testée sur une vraie IA** | Conçue (DESIGN §7), à valider sur de vraies réponses. |

## 5. Mesures nouvelles (prototype `look`, LUT 33³ réelle)

Cas du brief (`run_p1.py`) :

| Cas A — Tokyo vert | L moy. | Chroma | Px colorés (C ≥ 0,05) | Ombres, axe a | Repliements |
|---|---|---|---|---|---|
| Source | 0,426 | 0,031 | 13,8 % | +0,004 | — |
| Référence | 0,280 | 0,053 | 43,4 % | −0,032 | — |
| Actuel : zones + recette IA dosée à 100 % | 0,292 | 0,041 | 19,0 % | −0,022 | 0,0043 |
| Prototype `look` | 0,296 | 0,045 | 28,3 % | −0,031 | 0 |

| Cas B — nuit violette | L moy. | Chroma | Px colorés | Concentration de teinte | 95ᵉ centile de L |
|---|---|---|---|---|---|
| Référence | 0,273 | 0,117 | 80,4 % | 0,99 | 0,42 |
| Actuel : ambiance | 0,273 | 0,110 | 99,4 % | 0,94 | 0,53 |
| Prototype `look` (vives protégées à 0,8) | 0,301 | 0,105 | 88,4 % | 0,93 | 0,84 |

Lecture visuelle (`out/p1_case_A.jpg`, `out/p1_case_B.jpg`) : en A, ombres vert-noir
denses comme la référence, enseignes encore lumineuses ; en B, violet profond et saturé
au lieu d'un voile laiteux, blancs presque blancs, néons intacts.

Six autres paires (`run_p2.py`, `out/p2_pairs_*.jpg`), réglages par défaut, contre les
zones automatiques : `look` est plus propre ou plus proche sur quatre paires (1, 2, 3,
5), équivalent ou un peu moins franc sur la n° 4 (mur rouge / feuillage vert : les zones
donnent des couleurs plus tranchées), et **plus terne sur la n° 6** (source de jour
presque grise, chroma 0,018 contre 0,021). Aucun repliement sur les huit rendus.

## 6. Ce qui reste hors d'atteinte, quel que soit le moteur

- Traiter différemment deux régions de même couleur (ciel noir / façade noire) avec un
  seul `.cube`.
- Faire naître des couleurs variées d'une source presque grise : on peut amplifier des
  écarts de teinte, pas les inventer (paire n° 6).
- Atteindre le nombre de pixels très colorés de la référence quand la source en a deux
  fois moins (cas A : 13,8 % contre 23 %) sans saturer uniformément.
- Grain, halation, diffusion : effets spatiaux.

## 7. Limites de cet audit

- Huit paires, toutes urbaines, sept de nuit ; aucun portrait, aucun paysage de jour
  coloré, aucune peau. Les constantes du prototype sont des valeurs d'essai.
- Les jugements « meilleur / équivalent » ci-dessus ont été portés sur des planches
  réduites : **Mathieu doit trancher sur `out/`**.
- La recette d'IA du cas A est la seule réponse réelle disponible.
