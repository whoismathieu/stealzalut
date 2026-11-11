# Approche v3 — état d'avancement

Branche : `approach-v3` (partie de `main` au commit `fd49d2c`). Rien n'est poussé.

| Étape | Livrable | État |
|---|---|---|
| A. Audit | `docs/audit/AUDIT.md` | **fait** |
| B. Conception + prototypes | `DESIGN.md`, `prototypes/`, planches dans `prototypes/out/` | **fait** |
| C. Spécification | `SPEC.md`, `IMPLEMENTATION_PLAN.md`, `ACCEPTANCE.md` | **fait** |
| D. Implémentation | T01 moteur `core/look.py`, T02 enregistrement, T03 curseur « Séparation », T04 limites affichées, T05 zones de référence à ignorer ou privilégier, T08 « Rendu » par défaut | **fait** (411 tests, `ruff` et `mypy --strict` propres) |
| D. Implémentation | T06 recette `look`, T07 pack IA avec rendu | **à faire**, spécifié |
| — | garder ou retirer « Ambiance », phase 2 (masques à la sortie) | **décision de Mathieu** |

## Pour reprendre

1. Lire `DESIGN.md`, puis `SPEC.md` §0 (décidé / ouvert / interdit).
2. Prendre la première tâche « à faire » de `IMPLEMENTATION_PLAN.md` (T06).
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
