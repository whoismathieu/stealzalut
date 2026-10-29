# StealZaLut

> Application de bureau professionnelle permettant d'extraire la signature colorimétrique d'une image de référence, de l'appliquer à une image source, et d'exporter le transfert sous forme de profil 3D LUT (`.cube`) pour Lightroom, DaVinci Resolve, Photoshop, etc.

---

## 🚀 Fonctionnalités Clés

* **Transfert Colorimétrique par Transport Optimal (MKL) :** Calcul analytique fermé de la matrice de transport de Monge–Kantorovitch entre les distributions gaussiennes de couleurs (Wasserstein-2).
* **Moteur 3D LUT Dynamique :** Génération procédurale d'une grille 3D neutre en mémoire à résolution configurable (**17³, 33³, ou 64³** points) respectant strictement l'ordre d'itération standard `.cube` (R-rapide, G-moyen, B-lent).
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
* **Gestionnaire de Paquets :** `uv` (PEP 621 / `pyproject.toml`)
* **Configuration :** `python-dotenv` (`.env`)
* **Packaging :** Prêt pour `PyInstaller`

---

## ⚡ Démarrage Rapide

### 1. Installation des dépendances

```bash
# Avec uv sous WSL / Linux
UV_LINK_MODE=copy uv sync --all-extras

# Ou configuration du fichier local
cp .env.example .env
```

### 2. Lancement de l'application

```bash
# Via uv
UV_LINK_MODE=copy uv run stealzalut

# Ou directement avec le module Python
python -m stealzalut.main
```

---

## 🧪 Tests & Qualité de Code

```bash
# Tests unitaires et d'intégration (63 tests)
UV_LINK_MODE=copy uv run pytest tests/ -v

# Typage statique strict
UV_LINK_MODE=copy uv run mypy src/

# Linter et conformité PEP 8
UV_LINK_MODE=copy uv run ruff check .
```

---

## 📖 Documentation d'Architecture

Consultez [ARCHITECTURE.md](ARCHITECTURE.md) pour les diagrammes de séquence et de flux de données détaillant la séparation entre le `core`, les `workers`, et l'interface `gui`.
