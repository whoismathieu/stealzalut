# StealZaLut

> Application de bureau professionnelle permettant d'extraire la signature colorimétrique d'une image de référence, de l'appliquer à une image source, et d'exporter le transfert sous forme de profil 3D LUT universel (`.cube`).

---

## 🚀 Fonctionnalités Clés

* **Export 3D LUT (.cube) Professionnel :**
  * Exportez le fichier 3D LUT universel prêt à l'emploi pour DaVinci Resolve, Photoshop, Premiere Pro, Final Cut Pro, OBS.
  * Résolution configurable (**17³, 33³, ou 64³** points) respectant strictement l'ordre d'itération standard `.cube` (R-rapide, G-moyen, B-lent).

* **Développement RAW & DNG Photographiques Haute Intensité :**
  * **Extraction des aperçus haute résolution intégrés** générés par les boîtiers photo et smartphones (colorimétrie et profil constructeur natifs).
  * **Compensation d'exposition DNG `BaselineExposure`** automatique.
  * **Courbe tonale photographique et rehaussement de vibrance** automatique : élimine les images sombres et ternes issues du dématriçage brut linéaire pour restituer le contraste, l'intensité et la dynamique réels de la scène.

* **Multiples Algorithmes de Transfert Colorimétrique :**
  * **MKL (Monge-Kantorovitch Linear) :** Transport optimal analytique fermé entre distributions gaussiennes (Wasserstein-2).
  * **Reinhard :** Alignement statistique de moyenne et variance dans l'espace perceptuel $L\alpha\beta$ de Ruderman.
  * **Histogram Matching (HM) :** Égalisation et alignement précis des distributions cumulées par canal (CDF matching).
  * **MVGD :** Décalage multivarié gaussien.
  * **Hybrides (HM-MKL-HM & HM-MVGD-HM) :** Pipeline en 3 étapes combinant alignement d'histogramme non-linéaire et transport optimal pour des contrastes et teintes riches.
* **Moteur 3D LUT Dynamique & Cuisson Universelle :** Évaluation directe de l'algorithme sélectionné sur une grille 3D neutre en mémoire à résolution configurable (**17³, 33³, ou 64³** points) respectant strictement l'ordre d'itération standard `.cube` (R-rapide, G-moyen, B-lent).
* **Formats d'Images Supportés :**
  * **Standards :** JPEG, PNG, TIFF, BMP, WebP
  * **Apple / Mobile :** HEIC / HEIF / HIF (avec correction automatique de l'orientation EXIF via `pillow-heif`)
  * **RAW Boîtiers Professionnels & DNG :** Dématriçage haute fidélité via `rawpy` (LibRaw) avec respect de la balance des blancs du boîtier :
    * **Canon :** `.cr2`, `.cr3`, `.crw`
    * **Nikon :** `.nef`, `.nrw`
    * **Sony :** `.arw`, `.srf`, `.sr2`
    * **Fujifilm :** `.raf`
    * **Panasonic Lumix :** `.rw2`, `.raw`
    * **Olympus / OM System :** `.orf`
    * **Pentax :** `.pef`, `.ptx`
    * **Adobe / Leica / ProRAW :** `.dng`, `.rwl`
    * **Hasselblad / Phase One / Sigma :** `.3fr`, `.fff`, `.iiq`, `.x3f`
* **Interface Réactive PySide6 :** Glisser-déposer d'images (**drag-and-drop**), sélection interactive de la résolution LUT, aperçu en direct du rendu, et barre de progression continue.
* **Architecture Asynchrone QThread :** Découplage strict entre l'UI et les calculs lourds (aucun gel de l'interface lors des opérations matricielles).
* **Architecture Modulaire de Niveau Production :**
  * `core/` : calculs mathématiques purs sans dépendance Qt, 100 % testable unitairement.
  * `workers/` : encapsulation `QThread` avec signaux Qt pour la progression et les résultats.
  * `gui/` : widgets, layout réactif et feuille de style sombre (`theme.qss`).
  * `utils/` : lecture OpenCV sécurisée et abstraction des chemins pour compatibilité PyInstaller (`sys._MEIPASS`).

---

## 🛠️ Stack Technique

* **Langage :** Python 3.10+ (Typage strict `mypy`)
* **Interface Graphique :** `PySide6` (Qt 6)
* **Calculs & Couleurs :** `numpy`, `opencv-python`, `color-matcher`
* **Décodage d'Images Avancé :** `Pillow`, `pillow-heif` (HEIC), `rawpy` (DNG / RAW)
* **Gestionnaire de Paquets :** `uv` (PEP 621 / `pyproject.toml`)
* **Configuration :** `python-dotenv` (`.env`)
* **Packaging :** Prêt pour `PyInstaller`

---

## ⚡ Démarrage Rapide

### 1. Installation des dépendances

**Sous Windows (PowerShell / Terminal) :**
```powershell
# Synchronisation de l'environnement virtuel avec uv
uv sync --all-extras

# Configuration locale
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

**Sous WSL / Linux :**
```bash
UV_LINK_MODE=copy uv sync --all-extras
cp .env.example .env
```

### 2. Lancement de l'application

**Sous Windows :**
```powershell
# Méthode 1 : double-clic sur launch.bat (ou exécution de launch.bat / launch.ps1)
.\launch.bat

# Méthode 2 : via uv
uv run stealzalut

# Méthode 3 : directement avec l'environnement virtuel
.\.venv\Scripts\stealzalut-gui.exe
```

**Sous WSL / Linux :**
```bash
UV_LINK_MODE=copy uv run stealzalut
```

---

## 🛠️ Commandes Makefile (Développement)

Un [`Makefile`](file:///c:/Users/mathi/Documents/StealZaLut/Makefile) est disponible à la racine (utilisable dans le terminal, sous WSL/Linux, ou via le plugin Makefile de PyCharm) :

```bash
make help        # Affiche la liste des commandes disponibles
make format      # Formate le code (ruff format + check --fix)
make lint        # Vérifie le respect du style PEP 8 (ruff check)
make typecheck   # Vérification de typage strict (mypy)
make test        # Lance les tests unitaires (pytest)
make test-cov    # Lance les tests avec rapport de couverture
make check       # Validation complète : lint + typecheck + test
make clean       # Nettoie les caches (__pycache__, pytest, ruff, mypy...)
make run         # Lance l'application
make build       # Construit l'exécutable PyInstaller
```

---

## 🧪 Tests & Qualité de Code

**Sous Windows (PowerShell) :**
```powershell
# Tests unitaires et d'intégration (105 tests)
uv run pytest tests/ -v

# Typage statique strict
uv run mypy src/

# Linter et conformité PEP 8
uv run ruff check .
```

**Sous WSL / Linux :**
```bash
UV_LINK_MODE=copy uv run pytest tests/ -v
UV_LINK_MODE=copy uv run mypy src/
UV_LINK_MODE=copy uv run ruff check .
```

---

## 📖 Documentation d'Architecture

Consultez [ARCHITECTURE.md](ARCHITECTURE.md) pour les diagrammes de séquence et de flux de données détaillant la séparation entre le `core`, les `workers`, et l'interface `gui`.
