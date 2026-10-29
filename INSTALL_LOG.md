# Installation Log — StealZaLut

> Ce fichier répertorie toutes les commandes d'installation et les éléments
> créés sur le système, pour pouvoir annuler/nettoyer si nécessaire.

---

## 1. Installation de `uv` (gestionnaire de paquets Python)

**Commande exécutée :**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Ce qui a été installé :**
- Binaire `uv` → `/home/mathi/.local/bin/uv`
- Binaire `uvx` → `/home/mathi/.local/bin/uvx`
- Script env → `/home/mathi/.local/bin/env`

**Pour désinstaller uv :**
```bash
rm -f ~/.local/bin/uv ~/.local/bin/uvx ~/.local/bin/env
```

---

## 2. Installation des dépendances Python du projet

**Commande :**
```bash
cd /mnt/c/Users/mathi/Documents/StealZaLut
UV_LINK_MODE=copy uv sync --all-extras
```

**Ce que ça fait :**
- Crée un environnement virtuel local dans `.venv/` à la racine du projet
- Installe les dépendances listées dans `pyproject.toml` :
  - `PySide6` (interface graphique Qt)
  - `numpy` (calcul matriciel)
  - `opencv-python` (lecture d'images)
  - `color-matcher` (transfert colorimétrique)
  - `python-dotenv` (config .env)
  - Extras dev : `pytest`, `pytest-cov`, `ruff`, `mypy`
  - Extras build : `pyinstaller`
- Crée un fichier `uv.lock` à la racine du projet

**Pour tout nettoyer :**
```bash
# Supprimer l'environnement virtuel et le lockfile
rm -rf /mnt/c/Users/mathi/Documents/StealZaLut/.venv
rm -f  /mnt/c/Users/mathi/Documents/StealZaLut/uv.lock
```

---

## 3. Commandes de vérification (qualité & tests)

**Tests unitaires (pytest) :**
```bash
cd /mnt/c/Users/mathi/Documents/StealZaLut
UV_LINK_MODE=copy uv run pytest tests/ -v
```

**Vérification de typage strict (mypy) :**
```bash
cd /mnt/c/Users/mathi/Documents/StealZaLut
UV_LINK_MODE=copy uv run mypy src/
```

**Linter / formateur (ruff) :**
```bash
cd /mnt/c/Users/mathi/Documents/StealZaLut
UV_LINK_MODE=copy uv run ruff check .
```

*Ces commandes n'installent rien de plus sur le système, elles valident uniquement le code source dans l'environnement virtuel local.*

---

## 4. Lancement de l'application

**Sous WSL (avec WSLg / serveur X) :**
```bash
cd /mnt/c/Users/mathi/Documents/StealZaLut
UV_LINK_MODE=copy uv run stealzalut
```

**Sous Windows (directement si Python 3.10+ est installé) :**
```powershell
cd c:\Users\mathi\Documents\StealZaLut
python -m stealzalut.main
```

---

## Nettoyage complet (tout annuler)

```bash
# 1. Supprimer uv
rm -f ~/.local/bin/uv ~/.local/bin/uvx ~/.local/bin/env

# 2. Supprimer l'environnement virtuel du projet
rm -rf /mnt/c/Users/mathi/Documents/StealZaLut/.venv
rm -f  /mnt/c/Users/mathi/Documents/StealZaLut/uv.lock
```

