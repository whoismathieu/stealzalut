# Approche v3 — état d'avancement

Branche : `approach-v3` (partie de `main` au commit `fd49d2c`). Rien n'est poussé.

| Étape | Livrable | État |
|---|---|---|
| A. Audit | `docs/audit/AUDIT.md` | **fait** |
| B. Conception + prototypes | `docs/approach-v3/DESIGN.md`, `prototypes/` | prototypes faits, document en cours |
| C. Spécification | `SPEC.md`, `IMPLEMENTATION_PLAN.md`, `ACCEPTANCE.md` | à faire |
| D. Implémentation | code + tests | à faire |

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
