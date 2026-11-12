# Approche v3 — état d'avancement

> Version courte et à jour : `CONTEXT.md` à la racine du dépôt. Lire celui-là d'abord.

Branche : `approach-v3` (partie de `main` au commit `fd49d2c`). Rien n'est poussé.

| Étape | Livrable | État |
|---|---|---|
| A. Audit | `docs/audit/AUDIT.md` | **fait** |
| B. Conception + prototypes | `DESIGN.md`, `prototypes/`, planches dans `prototypes/out/` | **fait** |
| C. Spécification | `SPEC.md`, `IMPLEMENTATION_PLAN.md`, `ACCEPTANCE.md` | **fait** |
| D. Implémentation | T01 moteur `core/look.py`, T02 enregistrement, T03 curseur « Séparation », T04 limites affichées, T05 zones de référence à ignorer ou privilégier, T06 recette `.szl` en mode Rendu, T07 pack IA avec le rendu actuel, T08 « Rendu » par défaut | **fait** (443 tests, `ruff` et `mypy --strict` propres) |
| D. Validation | prompt du pack IA (T07) à essayer sur une vraie IA, cas A et B | **à faire par Mathieu** |
| D. Fidélité de l'entrée | T09 : profils couleur convertis en sRGB, bandeau pour les sources RAW, export de l'image gradée | **fait** (456 tests) ; l'accord avec Lightroom reste **à vérifier par Mathieu** |
| D. Espace du `.cube` | T10 : sRGB, Display P3, Adobe RGB, ProPhoto, Rec.709 γ 2,4 | **fait** (502 tests) ; accord avec Lightroom et Resolve **à vérifier par Mathieu** |
| D. Lecture des images | T10 ter : le moteur analyse les couleurs d'origine (Display P3, Adobe RGB, ProPhoto) sans écrêtage sRGB | **fait** (547 tests) |
| D bis. Chaîne hors sRGB | T10 quater : `apply_lab` natif pour `oklab`, `zones`, `ambiance` ; un seul chemin dans `build_grid` ; la source entre dans la LUT dans son espace (aperçu, export d'image) ; pack IA sur couleurs non écrêtées | **fait** (627 tests) |
| D ter. Moteur `look` | Étirement linéaire de chroma par bande ; protection des couleurs vives limitée aux teintes autres que celle de la référence | **fait** (637 tests). Même contenu : 0,11 → 0,07 (réglage), 0,12 → 0,06 (contrôle). Autre contenu inchangé. Détail dans `BENCH_BASELINE.md` |
| D quater. Localité | Curseur « Fidélité à ma photo » du mode `look` : limite douce du déplacement (`with_locality`, `reach`), dans l'historique, les recettes (`look.locality`) et le prompt du pack IA. Désactivé par défaut | **fait** (656 tests). À fond : autre contenu 4,2 → 1,45 (réglage), 6,3 → 1,6 (contrôle) ; même contenu 0,06 → 0,43. **Valeur utile à juger à l'œil par Mathieu** |
| E. Calibrage | T11 : banc à vérité connue, en espace large, sur deux lots. Mesure de départ sur 25 photos dans `BENCH_BASELINE.md` | **mesuré** (565 tests). Constat : même contenu bon (`look` 0,11), autre contenu pire que rien pour les trois méthodes. Recalibrage **suspendu** : le défaut n'est pas dans les constantes. Trois essais mesurés (photo moyenne et invariants rejetés ; transport local : sur-transfert divisé par 2 à 3, sans passer sous 1). Intégration d'un rayon de localité : **décision de Mathieu** |
| — | garder ou retirer « Ambiance », phase 2 (masques à la sortie) | **décision de Mathieu** |

## Pour reprendre

1. Lire `DESIGN.md`, puis `SPEC.md` §0 (décidé / ouvert / interdit).
2. Toutes les tâches T01 à T08 sont faites. Ce qui reste : la validation du prompt (T07) sur une vraie IA, puis les décisions de Mathieu.
3. Les numéros de ligne cités dans le plan datent du commit `098396b` ; T03 a décalé
   `gui/main_window.py` d'une quarantaine de lignes : chercher les symboles par leur nom.

## Pièges rencontrés pendant ce travail

- Une égalisation complète de l'histogramme de luminosité ternit les enseignes : garder
  le saut 85 % → 99,9 % de la courbe de tons.
- Sans compandage de la chroma, le treillis est trop grossier près du gris et la
  séparation disparaît.
- `weighted_quantile` doit écarter les poids nuls avant d'interpoler.
- Écrire un saut de ligne échappé (barre oblique inverse + n) dans un script de correctif passé par un *heredoc* a produit un vrai saut
  de ligne dans une chaîne Python : préférer l'outil d'édition.

## Non vérifié

- Rendu de l'export `.cube` dans Lightroom ou Resolve (seul l'aperçu interne, qui
  applique la même grille, a été contrôlé).
- Portraits, peau, paysages de jour colorés.
- Le prompt du mode `look` (T07) n'a été soumis à aucune IA.
- Jugement visuel : les comparaisons ont été faites sur des planches réduites.

## Hypothèses prises sans réponse de Mathieu

- `.cube` obligatoire ; sorties supplémentaires acceptées si optionnelles.
- Hôtes visés : Lightroom et Resolve.
- IA de l'utilisateur : une IA de chat avec vision, par fichiers et presse-papiers.

## Comment rejouer les prototypes

```bash
PYTHONPATH=src .venv/Scripts/python.exe docs/approach-v3/prototypes/run_p1.py <dossier_jetable>
PYTHONPATH=src .venv/Scripts/python.exe docs/approach-v3/prototypes/run_p2.py <dossier_jetable>
PYTHONPATH=src .venv/Scripts/python.exe docs/approach-v3/prototypes/run_p3.py <dossier_jetable>
PYTHONPATH=src .venv/Scripts/python.exe docs/approach-v3/prototypes/diag_loss.py
```

Les chemins des photos sont en dur dans `prototypes/harness.py` (dossier Images de
Mathieu). Les planches de référence sont dans `prototypes/out/`.
