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

**Sous Windows :**
```powershell
cd c:\Users\mathi\Documents\StealZaLut

# Via le script de lancement rapide (double-clic aussi possible)
.\launch.bat

# Ou via uv
uv run stealzalut

# Ou directement l'exécutable GUI du venv
.\.venv\Scripts\stealzalut-gui.exe
```

**Sous WSL (avec WSLg / serveur X) :**
```bash
cd /mnt/c/Users/mathi/Documents/StealZaLut
UV_LINK_MODE=copy uv run stealzalut
```

---

## 5. Installation sous Windows (Détails)

1. **`uv` sur Windows :**
   - Installé via `Python 3.10` dans `C:\Users\mathi\AppData\Local\Programs\Python\Python310\Scripts\uv.exe`.
   - Accessible directement en PowerShell / CMD car le dossier est dans le `PATH` utilisateur.

2. **Environnement virtuel Windows :**
   - Créé dans `.venv/` avec l'interpréteur Windows `Python 3.10`.
   - Packages natifs Windows (wheels `.whl`) installés : `PySide6` (Qt 6), `numpy`, `opencv-python`, `color-matcher`, etc.

3. **Scripts de lancement :**
   - `launch.bat` et `launch.ps1` à la racine pour un démarrage immédiat.

---

## Nettoyage complet (tout annuler)

**Sous Windows :**
```powershell
# Supprimer l'environnement virtuel et scripts
Remove-Item -Recurse -Force .venv
Remove-Item launch.bat, launch.ps1
```

**Sous Linux / WSL :**
```bash
# 1. Supprimer uv
rm -f ~/.local/bin/uv ~/.local/bin/uvx ~/.local/bin/env

# 2. Supprimer l'environnement virtuel du projet
rm -rf /mnt/c/Users/mathi/Documents/StealZaLut/.venv
rm -f  /mnt/c/Users/mathi/Documents/StealZaLut/uv.lock
```

