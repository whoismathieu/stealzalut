# Banc à vérité connue — mesure de départ

> Mesuré le 2026-11-12 au commit `1393651`, sur les 25 photos numériques neutres du
> dossier `benchmark_photos_source` de Mathieu (10 nuit, 5 rue, 10 paysage ; aucune
> ne porte de profil ICC). Les photos ne sont pas dans le dépôt.
>
> Commande : `python -m stealzalut.devtools.ground_truth <dossier>` (réglages par défaut :
> 8 étalonnages, 2 paires « autre contenu » par photo, LUT 33³ en Display P3).

## Lecture

**1. Même contenu : le mode `look` est le meilleur, et la mesure est stable.**
Résidu moyen 0,11 (réglage) et 0,12 (contrôle), contre 0,17 / 0,13 pour `zones` et
0,24 / 0,21 pour `oklab`. Les deux lots concordent : ce n'est pas un effet des photos
choisies. Points faibles de `look` : gain de saturation (0,23–0,28), sarcelle-orange
(0,22), dominante violette (0,14–0,18). Erreur absolue moyenne : 0,004.

**2. Autre contenu : les trois méthodes font bien pire que ne rien faire, à égalité.**
Résidu moyen 4,1 à 6,3 ; médian 2,3 à 3,8. Seuls **13 %** des cas font mieux que
laisser la source telle quelle. Les trois méthodes sont à égalité : ce n'est pas une
affaire de constantes, c'est commun à tout appariement de distributions.

**3. Pourquoi.** Erreur absolue moyenne de `look` sur un autre contenu : **0,156**,
dont l'essentiel en luminosité (0,11–0,17). Taille moyenne d'un étalonnage du banc :
**0,056**. Deux photos sans rapport diffèrent donc environ trois fois plus entre elles
que l'étalonnage ne les change : le signal cherché est noyé dans la différence de sujet.

**4. Rester dans la même catégorie n'aide pas.** Nuit → nuit, rue → rue, paysage →
paysage : résidu médian 3,0 / 3,6 / 3,1, comme entre catégories différentes (2,7).

| `look`, autre contenu | Cas | Résidu médian | Erreur L | Erreur couleur |
|---|---|---|---|---|
| même catégorie | 255 | 3,03 | 0,139 | 0,057 |
| catégories différentes | 143 | 2,71 | 0,134 | 0,047 |

**5. Ce que cette mesure exagère.** Le banc pose comme idéal « l'étalonnage seul » :
copier l'exposition ou la palette de la référence y compte comme une erreur, alors
qu'un utilisateur qui choisit une référence sombre veut souvent un rendu sombre. Et ses
étalonnages sont modérés (0,056) ; un rendu argentique marqué pèse plus lourd face aux
différences de sujet. Le défaut est réel (voir les planches du croisement : ciel bleu
devenu rouge brique, nuages verts), mais « pire que rien » décrit ce banc, pas
forcément l'usage.

## Essai négatif : lire la référence contre des statistiques moyennes

Idée testée : ajuster le modèle non pas entre la source et la référence, mais entre
des pixels tirés d'autres photos (une « photo moyenne », prise dans l'autre lot) et la
référence, puis l'appliquer à la source. Variante : moitié photo moyenne, moitié source.

| Lot réglage, `look` | Même contenu (moy. / méd.) | Autre contenu (moy. / méd.) | Erreur L | Erreur couleur |
|---|---|---|---|---|
| actuel | 0,11 / 0,06 | 4,12 / 2,41 | 0,109 | 0,050 |
| photo moyenne | 2,41 / 1,68 | 3,37 / 1,89 | 0,089 | 0,040 |
| moitié-moitié | 1,41 / 0,98 | 3,59 / 1,96 | 0,092 | 0,046 |

Le lot de contrôle donne les mêmes médianes (3,66 → 2,95 et 3,14). **Verdict : rejeté.**
Le gain sur un autre contenu est faible (on reste très au-dessus de 1) et le même
contenu est détruit. Une photo moyenne ne représente pas mieux le sujet de la référence
que la source ne le fait.

## Essai négatif : ne transférer que ce qui dépend peu du sujet

Script : `prototypes/bench_invariants.py`. Ton et couleur notés séparément ; erreur
moyenne rapportée à celle de ne rien faire, lot de réglage.

| Option | Même contenu | Autre contenu |
|---|---|---|
| Ton : points noir et blanc seulement | 0,41 | 1,72 |
| Ton : moitié de l'appariement | 0,51 | 1,44 |
| Ton : appariement complet (actuel) | 0,02 | 2,55 |
| Couleur : teinte de l'axe neutre par ton | 0,35 à 0,67 | 3,4 |
| Couleur : dominante moyenne par ton | 0,26 | 2,71 |
| Couleur : transport complet (actuel) | 0,10 | 3,00 |

**Verdict : rejeté.** Sur un autre contenu, aucune option ne fait mieux que ne rien
toucher. L'« axe neutre » (masse de couleur la plus dense à chaque ton) n'est pas un
invariant : dans un paysage, c'est le ciel. Les étalonnages du banc déplacent la couleur
de 0,018 en moyenne ; deux photos sans rapport diffèrent de 0,05 à 0,06.

## Essai positif, partiel : transport local

Script : `prototypes/bench_local.py`. Une couleur ne regarde que les couleurs de la
référence situées à moins d'un rayon ρ (noyau gaussien en Oklab) : le déplacement est
l'écart entre les centres locaux des deux densités, pondéré par la masse de référence
présente à cet endroit, itéré trois fois. Ce qui n'a pas de correspondant proche n'est
pas transporté. C'est l'idée du transport optimal relâché, appliquée localement.

| `look` actuel et local | Même contenu : ton / couleur | Autre contenu : ton / couleur | Autre contenu : mieux que rien |
|---|---|---|---|
| actuel (réglage) | 0,02 / 0,10 | 2,55 / 3,00 | 13 % des cas |
| ρ = 0,03 (réglage) | 0,85 / 0,50 | 1,01 / 1,32 | 43 % |
| ρ = 0,03 (contrôle) | 0,84 / 0,46 | 1,01 / 1,23 | 45 % |
| ρ = 0,06 (réglage) | 0,57 / 0,33 | 1,02 / 1,92 | 39 % |
| ρ = 0,06 (contrôle) | 0,59 / 0,32 | 1,09 / 2,00 | 35 % |
| ρ = 0,12 (réglage) | 0,31 / 0,31 | 1,35 / 2,41 | 26 % |

**Lecture.** Les deux lots concordent. Sur un autre contenu, le dommage est divisé par
deux à trois et la luminosité n'est presque plus abîmée ; mais on reste juste au-dessus
de 1 en moyenne : moins nocif, pas encore utile. Le prix est payé sur le même contenu,
surtout pour le ton : un étalonnage qui déplace les couleurs de plus que ρ sort du
rayon. ρ est un curseur entre « respecter la photo » et « coller à la référence » ; le
mode `look` actuel correspond à ρ infini.

**Limites de l'essai.** Mesuré sur le modèle directement, sans passer par une LUT, sur
2500 échantillons par image ; lent (plusieurs minutes par lot) ; la monotonie de la
luminosité n'est pas garantie comme dans `look`. Ce n'est pas du code de production.

## Conséquences

- **Ne pas recalibrer les constantes de `look` sur ce banc.** Le régime « même
  contenu » est déjà bon ; le régime « autre contenu » ne dépend pas d'elles.
- Le problème est l'**identifiabilité** : à partir d'une seule image, rien ne
  distingue l'étalonnage du sujet. Trois essais mesurés : photo moyenne (rejeté),
  invariants (rejeté), transport local (réduit le défaut sans le supprimer).
- **Piste ouverte : un rayon de localité dans le mode `look`**, exposé comme réglage
  (« respecter ma photo ↔ coller à la référence »). À décider avec Mathieu avant
  d'intégrer : le gain n'est pas net et le comportement par défaut changerait.
- Ce qu'aucun essai automatique ne fera : décider que « ce vert est de l'herbe, pas une
  dominante ». C'est un jugement sur le contenu ; le pont IA (T07) est le seul endroit
  du produit qui peut le porter.
- Le détecteur de référence « monochrome » se déclenche sur 12 des 25 paires croisées
  avec les références argentiques : seuil à revoir.

## Détail des mesures

Résidu = erreur de la méthode / erreur de ne rien faire (0 : étalonnage retrouvé ; 1 : inutile ; > 1 : pire que rien).

## Lot « reglage » : 13 images, 936 cas

| Méthode | Régime | Résidu moyen | Médian | Pire | Erreur L | Erreur couleur | Cas |
|---|---|---|---|---|---|---|---|
| look | même contenu | 0.11 | 0.06 | 0.72 | 0.001 | 0.004 | 104 |
| look | autre contenu | 4.12 | 2.41 | 28.86 | 0.109 | 0.050 | 208 |
| oklab | même contenu | 0.24 | 0.15 | 0.98 | 0.005 | 0.006 | 104 |
| oklab | autre contenu | 4.19 | 2.26 | 33.66 | 0.116 | 0.043 | 208 |
| zones | même contenu | 0.17 | 0.14 | 0.77 | 0.005 | 0.004 | 104 |
| zones | autre contenu | 4.30 | 2.30 | 33.64 | 0.116 | 0.049 | 208 |

Résidu moyen par étalonnage, même contenu :

| Étalonnage | look | oklab | zones |
|---|---|---|---|
| contraste | 0.05 | 0.32 | 0.32 |
| sombre | 0.02 | 0.10 | 0.09 |
| delave | 0.10 | 0.04 | 0.07 |
| sature | 0.23 | 0.24 | 0.27 |
| sarcelle-orange | 0.22 | 0.73 | 0.25 |
| nuit-verte | 0.07 | 0.17 | 0.11 |
| dominante-violette | 0.14 | 0.18 | 0.09 |
| chaud-doux | 0.07 | 0.13 | 0.12 |

Résidu moyen par étalonnage, autre contenu :

| Étalonnage | look | oklab | zones |
|---|---|---|---|
| contraste | 2.58 | 2.69 | 2.74 |
| sombre | 0.79 | 0.85 | 0.88 |
| delave | 3.14 | 3.20 | 3.26 |
| sature | 9.68 | 9.79 | 10.15 |
| sarcelle-orange | 10.01 | 9.98 | 10.23 |
| nuit-verte | 1.40 | 1.46 | 1.51 |
| dominante-violette | 2.39 | 2.48 | 2.52 |
| chaud-doux | 2.97 | 3.04 | 3.10 |

## Lot « controle » : 12 images, 855 cas

| Méthode | Régime | Résidu moyen | Médian | Pire | Erreur L | Erreur couleur | Cas |
|---|---|---|---|---|---|---|---|
| look | même contenu | 0.12 | 0.06 | 0.63 | 0.001 | 0.004 | 95 |
| look | autre contenu | 6.26 | 3.56 | 74.49 | 0.168 | 0.058 | 190 |
| oklab | même contenu | 0.21 | 0.13 | 0.95 | 0.005 | 0.005 | 95 |
| oklab | autre contenu | 5.83 | 3.77 | 44.24 | 0.162 | 0.055 | 190 |
| zones | même contenu | 0.13 | 0.09 | 0.48 | 0.005 | 0.003 | 95 |
| zones | autre contenu | 5.91 | 3.81 | 44.29 | 0.162 | 0.059 | 190 |

Résidu moyen par étalonnage, même contenu :

| Étalonnage | look | oklab | zones |
|---|---|---|---|
| contraste | 0.03 | 0.26 | 0.27 |
| sombre | 0.02 | 0.09 | 0.09 |
| delave | 0.12 | 0.03 | 0.04 |
| sature | 0.28 | 0.22 | 0.24 |
| sarcelle-orange | 0.22 | 0.68 | 0.18 |
| nuit-verte | 0.05 | 0.14 | 0.10 |
| dominante-violette | 0.18 | 0.16 | 0.04 |
| chaud-doux | 0.08 | 0.11 | 0.10 |

Résidu moyen par étalonnage, autre contenu :

| Étalonnage | look | oklab | zones |
|---|---|---|---|
| contraste | 4.04 | 3.81 | 3.85 |
| sombre | 1.15 | 1.15 | 1.16 |
| delave | 5.14 | 4.88 | 4.92 |
| sature | 15.93 | 13.96 | 14.18 |
| sarcelle-orange | 14.05 | 13.56 | 13.74 |
| nuit-verte | 2.01 | 1.96 | 1.99 |
| dominante-violette | 4.04 | 3.78 | 3.80 |
| chaud-doux | 4.54 | 4.25 | 4.31 |
