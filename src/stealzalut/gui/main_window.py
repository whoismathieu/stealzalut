"""Main application window.

Orchestrates the GUI layout and connects user interactions to the
background workers.  The window never performs heavy computation
directly — all matrix operations run in QThread workers.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from stealzalut.config import Settings
from stealzalut.core.ai_pack import build_prompt, write_pack
from stealzalut.core.color_transfer import METHOD_LABELS, TransferModel
from stealzalut.core.correspondence import CorrespondenceTable, Link
from stealzalut.core.grading import GridResult
from stealzalut.core.look import (
    SEPARATION_RANGE,
    LookDiagnostics,
    LookTransferModel,
    look_diagnostics,
)
from stealzalut.core.lut_engine import VALID_LUT_SIZES
from stealzalut.core.palette import (
    DEFAULT_ZONE_COUNT,
    MAX_ZONE_COUNT,
    MIN_ZONE_COUNT,
    ColourStats,
    colour_stats,
)
from stealzalut.core.recipe import (
    LOW_COLOUR_DOSE,
    RECIPE_SUFFIX,
    Recipe,
    RecipeError,
    apply_global,
    apply_recipe,
    colour_dose,
    dump_recipe,
    full_dose,
    merge_finishing,
    parse_recipe,
    recipe_to_dict,
)
from stealzalut.core.zone_transfer import (
    Conflict,
    ZoneBakeParams,
    ZoneTransferModel,
    find_conflicts,
)
from stealzalut.gui.history import EditState, History
from stealzalut.gui.widgets.adjustments_panel import AdjustmentsPanel
from stealzalut.gui.widgets.correspondence_view import SOURCE, CorrespondenceView
from stealzalut.gui.widgets.finishing_panel import FinishingPanel
from stealzalut.gui.widgets.image_drop_zone import ImageDropZone
from stealzalut.gui.widgets.paste_recipe_dialog import PasteRecipeDialog
from stealzalut.gui.widgets.preview_panel import PreviewPanel
from stealzalut.gui.widgets.zone_inspector import ZoneInspector
from stealzalut.gui.zone_text import dominant_colour_name
from stealzalut.workers.export_worker import ExportWorker
from stealzalut.workers.transfer_worker import TransferWorker

logger = logging.getLogger(__name__)

#: Delay between a correspondence edit and the preview refresh.
EDIT_DEBOUNCE_MS = 75
_TAU_STEPS = 100  # slider units per τ unit
_SEPARATION_STEPS = 100  # slider units per separation unit
_IDLE_MESSAGE = "Charge une source et une référence pour voir leurs zones."
_ANALYSING_MESSAGE = "Analyse des zones de couleur…"


class MainWindow(QMainWindow):
    """StealZaLut main application window.

    Layout::

        ┌──────────────────────────────────────────────────┐
        │  Source Image  │  Reference Image  │   Preview   │
        │  (drop zone)   │  (drop zone)      │   (result)  │
        ├──────────────────────────────────────────────────┤
        │  Correspondances (zones method only)             │
        │  S1 ═══▶ R2   …            │  Zone inspector     │
        ├──────────────────────────────────────────────────┤
        │  Ajustements (sliders)                           │
        ├──────────────────────────────────────────────────┤
        │  Algorithm │ LUT Size │ Undo/Redo │ [Export]     │
        ├──────────────────────────────────────────────────┤
        │  ████████████████░░░░░░░░  45%                   │
        └──────────────────────────────────────────────────┘
    """

    def __init__(self, settings: Settings, parent: Any = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._transfer_worker: TransferWorker | None = None
        self._export_worker: ExportWorker | None = None
        # Superseded preview workers still running; kept alive until they end.
        self._stale_transfer_workers: list[TransferWorker] = []
        # Fitted model reused while images and method are unchanged:
        # (source, reference, method, model).  Moving a slider or changing the
        # LUT size then only rebuilds the grid instead of re-reading the images.
        self._model_cache: tuple[Any, Any, str, TransferModel] | None = None
        self._pending_model_key: tuple[Any, Any, str] | None = None
        self._last_grid_result: GridResult | None = None
        # Zones method: raw fitted model shown in the editor (auto table)
        self._zone_model: ZoneTransferModel | None = None
        self._zone_count = DEFAULT_ZONE_COUNT
        # Recipe waiting for a re-analysis with its zone count
        self._pending_recipe: Recipe | None = None
        # Colour variety of the reference (suggests the ambiance mode)
        self._reference_colours: ColourStats | None = None
        self._mode_hint_dismissed = False
        # Known limits of the look method on the current pair
        self._look_diagnostics: LookDiagnostics | None = None
        # Message kept in the status bar when the next preview arrives
        self._status_note = ""

        self.setWindowTitle("StealZaLut — Colour LUT Generator")
        self.setMinimumSize(900, 680)
        self.resize(1280, 940)

        self._build_ui()
        self._history = History(EditState(None, self._adjustments.params(), self._tau()))
        self._current_method = str(self._algo_combo.currentData())

        # Correspondence edits refresh the preview once the user pauses
        self._edit_timer = QTimer(self)
        self._edit_timer.setSingleShot(True)
        self._edit_timer.setInterval(EDIT_DEBOUNCE_MS)
        self._edit_timer.timeout.connect(self._refresh_preview_if_ready)

        self._connect_signals()
        self._update_zone_panel_visibility()
        self._update_export_button_state()
        self._update_undo_buttons()
        self._update_ai_buttons()

        logger.info("MainWindow initialized")

    # ── UI Construction ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Build the complete widget tree."""
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(8)

        # ── Top: Image panels (source | reference | preview) ─────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Source drop zone
        source_group = QGroupBox("Source Image")
        source_layout = QVBoxLayout(source_group)
        self._source_zone = ImageDropZone(
            label_text="Drop SOURCE image\nor click to browse",
            max_preview=self._settings.preview_max_size,
        )
        source_layout.addWidget(self._source_zone)
        splitter.addWidget(source_group)

        # Reference drop zone
        ref_group = QGroupBox("Reference Image")
        ref_layout = QVBoxLayout(ref_group)
        self._reference_zone = ImageDropZone(
            label_text="Drop REFERENCE image\nor click to browse",
            max_preview=self._settings.preview_max_size,
        )
        ref_layout.addWidget(self._reference_zone)
        splitter.addWidget(ref_group)

        # Preview panel
        preview_group = QGroupBox("Transfer Preview")
        preview_layout = QVBoxLayout(preview_group)
        self._preview = PreviewPanel(title="")
        preview_layout.addWidget(self._preview)
        splitter.addWidget(preview_group)

        # Equal widths
        splitter.setSizes([350, 350, 350])

        # Images on top, correspondence editor below (resizable)
        self._main_splitter = QSplitter(Qt.Orientation.Vertical)
        self._main_splitter.setChildrenCollapsible(False)
        self._main_splitter.addWidget(splitter)
        self._corr_group = self._build_correspondence_panel()
        self._main_splitter.addWidget(self._corr_group)

        # ── Adjustments and finishing tools, in tabs ─────────────────
        self._bottom_tabs = QTabWidget()
        self._bottom_tabs.setObjectName("bottomTabs")
        self._adjustments = AdjustmentsPanel()
        self._bottom_tabs.addTab(self._adjustments, "Ajustements")
        self._finishing_panel = FinishingPanel()
        self._bottom_tabs.addTab(self._finishing_panel, "Finition")
        self._bottom_tabs.setTabToolTip(
            1, "Roues couleur, courbe de luminosité, réglages par couleur (après le transfert)"
        )
        self._main_splitter.addWidget(self._bottom_tabs)
        self._main_splitter.setSizes([300, 360, 250])
        self._mode_hint = self._build_mode_hint()
        root_layout.addWidget(self._mode_hint)
        root_layout.addWidget(self._main_splitter, stretch=1)

        # ── Middle: Controls (Algorithm + LUT size + Export button) ─
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(12)

        # Algorithm selector
        algo_label = QLabel("Algorithm:")
        algo_label.setObjectName("controlLabel")
        controls_layout.addWidget(algo_label)

        self._algo_combo = QComboBox()
        self._algo_combo.setObjectName("algoCombo")
        for method_key, method_name in METHOD_LABELS.items():
            self._algo_combo.addItem(method_name, userData=method_key)
        default_algo_idx = self._algo_combo.findData(self._settings.color_transfer_method)
        if default_algo_idx >= 0:
            self._algo_combo.setCurrentIndex(default_algo_idx)
        controls_layout.addWidget(self._algo_combo)

        # Separation of the look method (hidden for the other methods)
        separation_tooltip = (
            "Écarte les couleurs proches du gris vers les couleurs de la référence.\n"
            "0 : seulement la dominante de chaque ton. 1 : ce que montre la référence. "
            "Au-delà : accentué."
        )
        self._separation_label = QLabel("Séparation :")
        self._separation_label.setObjectName("controlLabel")
        self._separation_label.setToolTip(separation_tooltip)
        controls_layout.addWidget(self._separation_label)
        self._separation_slider = QSlider(Qt.Orientation.Horizontal)
        self._separation_slider.setObjectName("separationSlider")
        self._separation_slider.setRange(
            round(SEPARATION_RANGE[0] * _SEPARATION_STEPS),
            round(SEPARATION_RANGE[1] * _SEPARATION_STEPS),
        )
        self._separation_slider.setValue(_SEPARATION_STEPS)
        self._separation_slider.setFixedWidth(140)
        self._separation_slider.setToolTip(separation_tooltip)
        controls_layout.addWidget(self._separation_slider)
        self._separation_value = QLabel()
        self._separation_value.setObjectName("adjustValue")
        self._separation_value.setMinimumWidth(36)
        controls_layout.addWidget(self._separation_value)
        self._refresh_separation_label()

        # LUT size selector
        size_label = QLabel("LUT Size:")
        size_label.setObjectName("controlLabel")
        controls_layout.addWidget(size_label)

        self._size_combo = QComboBox()
        self._size_combo.setObjectName("lutSizeCombo")
        for size in sorted(VALID_LUT_SIZES):
            label = f"{size}  ({size**3:,} points)"
            self._size_combo.addItem(label, userData=size)
        # Set default from settings
        default_idx = self._size_combo.findData(self._settings.lut_default_size)
        if default_idx >= 0:
            self._size_combo.setCurrentIndex(default_idx)
        controls_layout.addWidget(self._size_combo)

        controls_layout.addStretch()

        # Undo / redo (correspondences and sliders)
        self._undo_btn = QPushButton("↶ Annuler")
        self._undo_btn.setObjectName("undoButton")
        self._undo_btn.setToolTip("Annuler la dernière modification (Ctrl+Z)")
        controls_layout.addWidget(self._undo_btn)
        self._redo_btn = QPushButton("↷ Rétablir")
        self._redo_btn.setObjectName("redoButton")
        self._redo_btn.setToolTip("Rétablir la modification annulée (Ctrl+Y)")
        controls_layout.addWidget(self._redo_btn)

        # Clear button
        self._clear_btn = QPushButton("Clear All")
        self._clear_btn.setObjectName("clearButton")
        controls_layout.addWidget(self._clear_btn)

        # Export button (.cube)
        self._export_btn = QPushButton("Export .cube")
        self._export_btn.setObjectName("exportButton")
        self._export_btn.setEnabled(False)
        self._export_btn.setToolTip("Exporter le fichier 3D LUT (.cube)")
        controls_layout.addWidget(self._export_btn)

        root_layout.addLayout(controls_layout)

        # ── Bottom: Progress bar ─────────────────────────────────────
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setObjectName("mainProgressBar")
        root_layout.addWidget(self._progress_bar)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready — drop a source and reference image to begin.")

    def _build_correspondence_panel(self) -> QGroupBox:
        """Correspondence editor: toolbar, conflict banner, view and inspector."""
        group = QGroupBox("Correspondances entre zones")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        zones_label = QLabel("Zones :")
        zones_label.setObjectName("controlLabel")
        toolbar.addWidget(zones_label)
        self._zone_count_spin = QSpinBox()
        self._zone_count_spin.setObjectName("zoneCountSpin")
        self._zone_count_spin.setRange(MIN_ZONE_COUNT, MAX_ZONE_COUNT)
        self._zone_count_spin.setValue(DEFAULT_ZONE_COUNT)
        self._zone_count_spin.setKeyboardTracking(False)
        self._zone_count_spin.setToolTip(
            "Nombre de zones de couleur par image.\n"
            "Plus de zones = réglage plus fin, mais plus de liens à vérifier.\n"
            "Changer ce nombre relance l'analyse."
        )
        toolbar.addWidget(self._zone_count_spin)

        tau_tooltip = (
            "Comment les zones se fondent entre elles dans la LUT.\n"
            "À gauche : chaque zone garde nettement sa propre couleur.\n"
            "À droite : transitions plus douces entre zones voisines."
        )
        tau_label = QLabel("Transitions :")
        tau_label.setObjectName("controlLabel")
        tau_label.setToolTip(tau_tooltip)
        toolbar.addWidget(tau_label)
        self._tau_slider = QSlider(Qt.Orientation.Horizontal)
        self._tau_slider.setObjectName("tauSlider")
        self._tau_slider.setRange(25, 300)
        self._tau_slider.setValue(round(ZoneBakeParams().tau * _TAU_STEPS))
        self._tau_slider.setFixedWidth(160)
        self._tau_slider.setToolTip(tau_tooltip)
        toolbar.addWidget(self._tau_slider)
        self._tau_value = QLabel()
        self._tau_value.setObjectName("adjustValue")
        self._tau_value.setMinimumWidth(36)
        toolbar.addWidget(self._tau_value)
        self._refresh_tau_label()

        toolbar.addStretch()

        # Bridge with the user's own AI (no network: files and clipboard only)
        self._ai_pack_btn = QPushButton("Préparer pour mon IA…")
        self._ai_pack_btn.setObjectName("aiPackButton")
        self._ai_pack_btn.setToolTip(
            "Crée un dossier à donner à ton IA (Claude, ChatGPT, Gemini…) :\n"
            "les zones en images, leurs mesures et le prompt à coller.\n"
            "Son conseil revient ensuite par « Importer »."
        )
        toolbar.addWidget(self._ai_pack_btn)

        self._import_btn = QPushButton("Importer ▾")
        self._import_btn.setObjectName("importRecipeButton")
        self._import_btn.setToolTip("Appliquer la réponse de ton IA ou une recette enregistrée")
        # Popped up by hand: QPushButton.setMenu() makes Qt warn about a
        # negative point size with the stylesheet (fonts are sized in px).
        self._import_menu = QMenu(self)
        self._paste_action = self._import_menu.addAction("Coller la réponse de l'IA…")
        self._open_recipe_action = self._import_menu.addAction("Ouvrir une recette (.szl)…")
        self._import_btn.clicked.connect(
            lambda: self._import_menu.popup(
                self._import_btn.mapToGlobal(self._import_btn.rect().bottomLeft())
            )
        )
        toolbar.addWidget(self._import_btn)

        self._export_recipe_btn = QPushButton("Exporter la recette…")
        self._export_recipe_btn.setObjectName("exportRecipeButton")
        self._export_recipe_btn.setToolTip(
            "Enregistre tes correspondances et réglages dans un fichier .szl\n"
            "(pour les reprendre plus tard ou les partager)."
        )
        toolbar.addWidget(self._export_recipe_btn)

        self._all_auto_btn = QPushButton("Tout remettre en auto")
        self._all_auto_btn.setObjectName("allAutoButton")
        self._all_auto_btn.setToolTip("Revenir à l'appariement automatique pour toutes les zones")
        toolbar.addWidget(self._all_auto_btn)
        layout.addLayout(toolbar)

        self._conflict_banner = QLabel()
        self._conflict_banner.setObjectName("conflictBanner")
        self._conflict_banner.setWordWrap(True)
        self._conflict_banner.hide()
        layout.addWidget(self._conflict_banner)

        body = QHBoxLayout()
        body.setSpacing(8)
        self._corr_view = CorrespondenceView()
        self._corr_view.clear(_IDLE_MESSAGE)
        view_scroll = QScrollArea()
        view_scroll.setObjectName("correspondenceScroll")
        view_scroll.setWidgetResizable(True)
        view_scroll.setFrameShape(QFrame.Shape.NoFrame)
        view_scroll.setWidget(self._corr_view)
        body.addWidget(view_scroll, stretch=1)

        self._inspector = ZoneInspector()
        inspector_scroll = QScrollArea()
        inspector_scroll.setObjectName("inspectorScroll")
        inspector_scroll.setWidgetResizable(True)
        inspector_scroll.setFrameShape(QFrame.Shape.NoFrame)
        inspector_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inspector_scroll.setWidget(self._inspector)
        inspector_scroll.setFixedWidth(370)
        body.addWidget(inspector_scroll)
        layout.addLayout(body, stretch=1)
        return group

    # ── Signal wiring ────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        """Wire up all signal-slot connections."""
        # Image loading → auto-preview
        self._source_zone.image_loaded.connect(self._on_image_changed)
        self._reference_zone.image_loaded.connect(self._on_image_changed)
        self._reference_zone.image_loaded.connect(self._on_reference_loaded)

        # Image cleared
        self._source_zone.image_cleared.connect(self._on_image_cleared)
        self._reference_zone.image_cleared.connect(self._on_image_cleared)

        # Buttons
        self._export_btn.clicked.connect(self._on_export_clicked)
        self._clear_btn.clicked.connect(self._on_clear_all)

        # Algorithm or LUT size changed → the previewed grid changes too
        self._algo_combo.currentIndexChanged.connect(self._on_algo_changed)
        self._size_combo.currentIndexChanged.connect(self._on_size_changed)

        # Slider moved (debounced by the panel) → rebuild the grid
        self._adjustments.params_changed.connect(self._on_params_changed)
        self._finishing_panel.finishing_changed.connect(self._on_finishing_changed)

        # Correspondence editor
        self._corr_view.table_changed.connect(self._on_table_edited)
        self._corr_view.zone_selected.connect(self._on_zone_selected)
        self._corr_view.zone_hovered.connect(self._on_zone_hovered)
        self._inspector.link_changed.connect(self._on_link_edited)
        self._zone_count_spin.valueChanged.connect(self._on_zone_count_changed)
        self._tau_slider.valueChanged.connect(self._on_tau_changed)
        self._separation_slider.valueChanged.connect(self._on_separation_changed)
        self._all_auto_btn.clicked.connect(self._on_all_auto)
        self._ai_pack_btn.clicked.connect(self._on_prepare_ai_pack)
        self._paste_action.triggered.connect(self._on_paste_recipe)
        self._open_recipe_action.triggered.connect(self._on_open_recipe)
        self._export_recipe_btn.clicked.connect(self._on_export_recipe)

        # Undo / redo
        self._undo_btn.clicked.connect(self.undo)
        self._redo_btn.clicked.connect(self.redo)
        QShortcut(QKeySequence.StandardKey.Undo, self, self.undo)
        QShortcut(QKeySequence.StandardKey.Redo, self, self.redo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self.redo)

    # ── Slots ────────────────────────────────────────────────────────────

    def _on_algo_changed(self, _index: int) -> None:
        """Called when the user changes the algorithm."""
        method = str(self._algo_combo.currentData())
        if method == self._current_method:
            return
        if not self._confirm_discard_edits("Changer de méthode"):
            self._algo_combo.blockSignals(True)
            self._algo_combo.setCurrentIndex(self._algo_combo.findData(self._current_method))
            self._algo_combo.blockSignals(False)
            return
        self._current_method = method
        self._look_diagnostics = None
        self._drop_zone_editor()
        self._update_zone_panel_visibility()
        self._update_mode_hint()
        self._refresh_preview_if_ready()

    def _on_size_changed(self, _index: int) -> None:
        """Called when the user changes the LUT size."""
        self._refresh_preview_if_ready()

    def _on_params_changed(self, params: Any) -> None:
        """Called (debounced) when an adjustment slider moves."""
        changed = self._history.push(replace(self._history.current, params=params), "params")
        self._update_undo_buttons()
        if changed:
            self._refresh_preview_if_ready()

    def _on_finishing_changed(self, finishing: Any, merge_key: str) -> None:
        """Called on every finishing edit (the preview refresh is debounced)."""
        state = replace(self._history.current, finishing=finishing)
        if self._history.push(state, merge_key or None):
            self._edit_timer.start()
        self._update_undo_buttons()

    def _on_image_changed(self, _image: Any) -> None:
        """Called when either source or reference image is loaded."""
        self._model_cache = None
        self._look_diagnostics = None
        self._mode_hint_dismissed = False
        self._drop_zone_editor()
        self._update_export_button_state()
        self._refresh_preview_if_ready()

    def _on_image_cleared(self) -> None:
        """Called when an image is removed."""
        self._look_diagnostics = None
        self._update_mode_hint()
        self._model_cache = None
        self._drop_zone_editor()
        self._update_export_button_state()
        self._preview.clear()
        self._progress_bar.setValue(0)

    def _on_clear_all(self) -> None:
        """Clear both images and the preview."""
        self._source_zone.clear_image()
        self._reference_zone.clear_image()
        self._preview.clear()
        self._progress_bar.setValue(0)
        self._status_bar.showMessage("Cleared — ready for new images.")

    def _on_export_clicked(self) -> None:
        """Open a save dialog and start the .cube export worker."""
        if not self._source_zone.has_image or not self._reference_zone.has_image:
            return

        # Determine default filename
        default_name = "stealzalut_export.cube"
        if self._source_zone.image_path and self._reference_zone.image_path:
            src_stem = self._source_zone.image_path.stem
            ref_stem = self._reference_zone.image_path.stem
            default_name = f"{src_stem}_x_{ref_stem}.cube"

        default_dir = str(self._settings.lut_output_dir)
        default_path = str(Path(default_dir) / default_name)

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export 3D LUT (.cube)",
            default_path,
            "Cube LUT (*.cube)",
        )
        if not path:
            return

        self._run_export(Path(path))

    # ── Correspondence editor ────────────────────────────────────────────

    def _on_table_edited(self, table: CorrespondenceTable, merge_key: str = "") -> None:
        """A table edit from the view or the inspector."""
        if self._zone_model is None:
            return
        state = replace(self._history.current, table=table)
        if self._history.push(state, merge_key or None):
            self._show_editor_state()
            self._edit_timer.start()
        self._update_undo_buttons()

    def _on_link_edited(self, link: Link, merge_key: str) -> None:
        table = self._history.current.table
        if table is not None:
            self._on_table_edited(table.with_link(link), merge_key)

    # ── Bridge with the user's AI and recipe files ───────────────────────

    def _on_prepare_ai_pack(self) -> None:
        """Write the analysis pack and copy the prompt to the clipboard."""
        model = self._zone_model
        source, reference = self._source_zone.image_rgb, self._reference_zone.image_rgb
        if model is None or source is None or reference is None:
            return
        destination = QFileDialog.getExistingDirectory(
            self, "Dossier où créer le pack pour ton IA", str(self._settings.lut_output_dir)
        )
        if not destination:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            folder = write_pack(
                Path(destination), self._base_name(), source, reference, model, self._zone_count
            )
        except OSError as exc:
            QMessageBox.critical(self, "Pack non créé", f"Impossible d'écrire le pack :\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        QGuiApplication.clipboard().setText(build_prompt(model))
        self._status_bar.showMessage(f"Pack pour ton IA créé : {folder}")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Pack prêt pour ton IA")
        box.setText(f"Pack créé dans :\n{folder}\n(et une copie .zip à côté)")
        box.setInformativeText(
            "1. Ouvre ton IA (Claude, ChatGPT, Gemini…).\n"
            "2. Joins les 6 images et analysis.json (ou le .zip) : les deux premières sont\n"
            "   tes photos telles quelles, pour que l'IA juge leur vraie ambiance.\n"
            "3. Colle le prompt : il est déjà dans ton presse-papiers (copie dans PROMPT.md).\n"
            "4. Copie toute sa réponse, puis ici : Importer › Coller la réponse de l'IA."
        )
        open_btn = box.addButton("Ouvrir le dossier", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Ok)
        box.exec()
        if box.clickedButton() is open_btn:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _on_paste_recipe(self) -> None:
        if not self._require_zones():
            return
        dialog = PasteRecipeDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.recipe is not None:
            self.apply_recipe(dialog.recipe)

    def _on_open_recipe(self) -> None:
        if not self._require_zones():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Ouvrir une recette",
            str(self._settings.lut_output_dir),
            f"Recette StealZaLut (*{RECIPE_SUFFIX});;JSON (*.json);;Tous les fichiers (*.*)",
        )
        if not path:
            return
        try:
            recipe = parse_recipe(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, RecipeError) as exc:
            QMessageBox.critical(self, "Recette illisible", str(exc))
            return
        self.apply_recipe(recipe)

    def _on_export_recipe(self) -> None:
        model, table = self._zone_model, self._history.current.table
        if model is None or table is None:
            return
        default = Path(self._settings.lut_output_dir) / f"{self._base_name()}{RECIPE_SUFFIX}"
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter la recette", str(default), f"Recette StealZaLut (*{RECIPE_SUFFIX})"
        )
        if not path:
            return
        state = self._history.current
        data = recipe_to_dict(
            table,
            state.params,
            state.tau,
            model.analysis_id,
            self._zone_count,
            name=Path(path).stem,
            finishing=state.finishing,
        )
        try:
            Path(path).write_text(dump_recipe(data), encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Recette non enregistrée", str(exc))
            return
        self._status_bar.showMessage(f"Recette enregistrée : {path}")

    def apply_recipe(self, recipe: Recipe, check_analysis: bool = True) -> bool:
        """Apply *recipe* to the current zones (one undo step).

        Returns:
            ``True`` if applied (or scheduled after a re-analysis).
        """
        if recipe.mode == "ambiance":
            return self._apply_ambiance_recipe(recipe)
        model = self._zone_model
        if model is None:
            return False
        if check_analysis and recipe.analysis_id and recipe.analysis_id != model.analysis_id:
            choice = self._ask_other_analysis(recipe)
            if choice == "cancel":
                return False
            if choice == "recompute" and recipe.zone_count is not None:
                self._pending_recipe = recipe
                self._zone_count_spin.setValue(recipe.zone_count)
                if self._zone_count != recipe.zone_count:  # change refused
                    self._pending_recipe = None
                    return False
                return True
        try:
            table = apply_recipe(recipe, model.table, model.reference.ids)
        except RecipeError as exc:
            QMessageBox.critical(self, "Recette non applicable", str(exc))
            return False
        params, tau = apply_global(recipe, self._history.current.params, self._history.current.tau)
        # Only the finishing settings the recipe gives replace the current ones
        # (an AI sets saturation / vibrance, the user's wheels and curve stay)
        finishing = merge_finishing(recipe, self._history.current.finishing)
        separation = self._history.current.separation
        state = EditState(table, params, tau, finishing, separation)
        if self._history.push(state):
            self._restore(state)
        title = f"« {recipe.name} »" if recipe.name else "importée"
        note = f"Recette {title} appliquée : {len(recipe.links)} zone(s) réglée(s)."
        boosts = [
            f"{label} {round(getattr(finishing, key) * 100):+d} %"
            for key, label in (("vibrance", "vibrance"), ("saturation", "saturation"))
            if key in recipe.finishing_keys and getattr(finishing, key) != 0.0
        ]
        if boosts and recipe.finishing_keys <= {"saturation", "vibrance"}:
            note += f" Finition : {', '.join(boosts)}."

        # Doses multiply: a recipe cautious everywhere barely changes anything
        areas = {zone.id: zone.area for zone in model.source.zones}
        dose = colour_dose(table, params, areas)
        if dose < LOW_COLOUR_DOSE and self._ask_low_dose(dose, table, params):
            full_table, full_params = full_dose(table, params)
            boosted = EditState(full_table, full_params, tau, finishing, separation)
            if self._history.push(boosted):  # its own undo step: Ctrl+Z = recipe dose
                self._restore(boosted)
            note += " Dosée à 100 % (Ctrl+Z pour revenir au dosage de la recette)."

        # Shown now and again when the re-baked preview arrives (not overwritten by it)
        self._status_note = f"{note} {recipe.description}".strip()
        self._status_bar.showMessage(self._status_note)
        return True

    def _apply_ambiance_recipe(self, recipe: Recipe) -> bool:
        """A recipe asking for the ambiance mode: switch, then apply its settings."""
        self._algo_combo.setCurrentIndex(self._algo_combo.findData("ambiance"))
        if self._current_method != "ambiance":  # switch refused (hand-edited links)
            return False
        current = self._history.current
        params, tau = apply_global(recipe, current.params, current.tau)
        finishing = merge_finishing(recipe, current.finishing)
        state = EditState(None, params, tau, finishing, current.separation)
        if self._history.push(state):
            self._restore(state)
        title = f"« {recipe.name} »" if recipe.name else "importée"
        note = (
            f"Recette {title} appliquée en mode Ambiance : la référence est lue comme une "
            "dominante par tons, les couleurs vives de la photo sont gardées."
        )
        self._status_note = f"{note} {recipe.description}".strip()
        self._status_bar.showMessage(self._status_note)
        return True

    # ── Monochrome reference hint ────────────────────────────────────────

    def _build_mode_hint(self) -> QFrame:
        """Banner suggesting the ambiance mode for a one-colour reference."""
        bar = QFrame()
        bar.setObjectName("modeHintBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 6, 6, 6)
        self._mode_hint_label = QLabel()
        self._mode_hint_label.setObjectName("modeHintLabel")
        self._mode_hint_label.setWordWrap(True)
        layout.addWidget(self._mode_hint_label, stretch=1)
        self._mode_hint_btn = QPushButton("Passer en Ambiance")
        self._mode_hint_btn.setObjectName("modeHintButton")
        self._mode_hint_btn.clicked.connect(
            lambda: self._algo_combo.setCurrentIndex(self._algo_combo.findData("ambiance"))
        )
        layout.addWidget(self._mode_hint_btn)
        close = QToolButton()
        close.setText("×")
        close.setObjectName("modeHintClose")
        close.setToolTip("Masquer")
        close.clicked.connect(self._dismiss_mode_hint)
        layout.addWidget(close)
        bar.hide()
        return bar

    def _on_reference_loaded(self, image: Any) -> None:
        self._reference_colours = colour_stats(image)
        self._mode_hint_dismissed = False
        self._update_mode_hint()

    def _dismiss_mode_hint(self) -> None:
        self._mode_hint_dismissed = True
        self._update_mode_hint()

    def _look_hint_text(self) -> str:
        """The limit of the look method worth telling the user, if any."""
        diagnostics = self._look_diagnostics
        if diagnostics is None:
            return ""
        if diagnostics.low_colour:
            return "La photo a peu de couleurs : la séparation aura peu d'effet."
        if diagnostics.weak_bands:
            return (
                "La référence ne montre presque rien dans certains tons : "
                "ils gardent leurs couleurs."
            )
        return ""

    def _update_mode_hint(self) -> None:
        if self._current_method == "look":
            # The look method protects vivid colours itself: no ambiance hint,
            # only its own limits.
            text = self._look_hint_text()
            self._mode_hint_btn.hide()
            self._mode_hint_label.setText(text)
            self._mode_hint.setVisible(bool(text) and not self._mode_hint_dismissed)
            return
        self._mode_hint_btn.show()
        stats = self._reference_colours if self._reference_zone.has_image else None
        show = (
            stats is not None
            and stats.is_monochrome
            and self._current_method != "ambiance"
            and not self._mode_hint_dismissed
        )
        if show and stats is not None:
            self._mode_hint_label.setText(
                f"La référence est dominée par une seule couleur ({dominant_colour_name(stats.hue)}"
                "). Avec cette méthode, toute la photo prendrait cette couleur, comme un filtre. "
                "Le mode Ambiance l'ajoute comme une dominante par tons et garde la couleur "
                "des néons, enseignes et lumières."
            )
        self._mode_hint.setVisible(show)

    def _ask_low_dose(self, dose: float, table: CorrespondenceTable, params: Any) -> bool:
        """Offer to raise a weak recipe to full dose; ``True`` = raise it."""
        unlocked = [link for link in table.links if not link.locked and link.targets]
        strengths = sorted({round(link.strength * 100) for link in unlocked})
        causes = [f"force par zone {', '.join(f'{s} %' for s in strengths)}"]
        if params.chroma_strength < 1.0:
            causes.append(f"intensité couleur {round(params.chroma_strength * 100)} %")
        if params.mix < 1.0:
            causes.append(f"mix {round(params.mix * 100)} %")
        if params.preserve_neutrals > 0.0:
            causes.append(
                f"préserver les neutres {round(params.preserve_neutrals * 100)} % "
                "(bride encore les tons peu saturés)"
            )
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Recette très peu dosée")
        box.setText(
            f"Avec cette recette, la couleur de la référence n'est appliquée qu'à "
            f"{round(dose * 100)} % en moyenne : ses réglages se multiplient\n"
            f"({' × '.join(causes)})."
        )
        box.setInformativeText(
            "« Doser à 100 % » garde les correspondances, les intentions et les zones "
            "protégées de la recette, et met les forces, l'intensité couleur et le mix à "
            "100 % (préserver les neutres à 0 %).\nCtrl+Z revient au dosage de la recette."
        )
        boost = box.addButton("Doser à 100 %", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Garder la recette", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(boost)
        box.exec()
        return box.clickedButton() is boost

    def _ask_other_analysis(self, recipe: Recipe) -> str:
        """``"apply"``, ``"recompute"`` or ``"cancel"`` for a recipe of other zones."""
        assert self._zone_model is not None
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Recette d'une autre analyse")
        text = (
            "Cette recette a été écrite pour d'autres zones "
            f"(analyse {recipe.analysis_id}, ici {self._zone_model.analysis_id}).\n"
            "Les numéros de zones ne désignent peut-être plus les mêmes choses."
        )
        recompute = None
        k = recipe.zone_count
        if k is not None and k != self._zone_count and MIN_ZONE_COUNT <= k <= MAX_ZONE_COUNT:
            text += f"\n\nElle a été faite avec {k} zones (ici {self._zone_count})."
            recompute = box.addButton(
                f"Recalculer avec {k} zones", QMessageBox.ButtonRole.AcceptRole
            )
        box.setText(text)
        apply_btn = box.addButton("Appliquer quand même", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if recompute is not None and clicked is recompute:
            return "recompute"
        if clicked is apply_btn:
            return "apply"
        return "cancel"

    def _require_zones(self) -> bool:
        if self._zone_model is not None:
            return True
        QMessageBox.information(
            self,
            "Zones pas encore prêtes",
            "Choisis la méthode « Zones » et charge une source et une référence : "
            "la recette s'applique à leurs zones.",
        )
        return False

    def _update_ai_buttons(self) -> None:
        ready = self._zone_model is not None
        for button in (self._ai_pack_btn, self._import_btn, self._export_recipe_btn):
            button.setEnabled(ready)

    def _base_name(self) -> str:
        """``<source>_x_<reference>`` for default file names."""
        src, ref = self._source_zone.image_path, self._reference_zone.image_path
        if src is not None and ref is not None:
            return f"{src.stem}_x_{ref.stem}"
        return "stealzalut"

    def _on_all_auto(self) -> None:
        if self._zone_model is not None:
            self._on_table_edited(self._zone_model.table)

    def _on_zone_selected(self, _zone_id: int) -> None:
        self._refresh_inspector()

    def _on_zone_hovered(self, side: str, zone_id: int) -> None:
        """Highlight the hovered zone on the matching images."""
        model = self._zone_model
        source_mask = reference_mask = None
        if model is not None and zone_id:
            if side == SOURCE:
                source_mask = model.source.labels == zone_id
            else:
                reference_mask = model.reference.labels == zone_id
        self._source_zone.set_highlight(source_mask)
        self._preview.set_highlight(source_mask)
        self._reference_zone.set_highlight(reference_mask)

    def _on_zone_count_changed(self, value: int) -> None:
        if value == self._zone_count:
            return
        if not self._confirm_discard_edits("Changer le nombre de zones"):
            self._zone_count_spin.blockSignals(True)
            self._zone_count_spin.setValue(self._zone_count)
            self._zone_count_spin.blockSignals(False)
            return
        self._zone_count = value
        self._drop_zone_editor()
        self._refresh_preview_if_ready()

    def _on_tau_changed(self, _value: int) -> None:
        self._refresh_tau_label()
        if self._history.push(replace(self._history.current, tau=self._tau()), "tau"):
            self._update_conflicts()
            self._edit_timer.start()
        self._update_undo_buttons()

    def _on_separation_changed(self, _value: int) -> None:
        self._refresh_separation_label()
        state = replace(self._history.current, separation=self._separation())
        if self._history.push(state, "separation"):
            self._edit_timer.start()
        self._update_undo_buttons()

    def undo(self) -> None:
        """Undo the last correspondence or slider change."""
        state = self._history.undo()
        if state is not None:
            self._restore(state)

    def redo(self) -> None:
        """Redo the last undone change."""
        state = self._history.redo()
        if state is not None:
            self._restore(state)

    def _restore(self, state: EditState) -> None:
        """Show *state* everywhere (after undo/redo)."""
        # The panel re-emits these params later; the history ignores them (no change)
        self._adjustments.set_params(state.params)
        self._finishing_panel.set_finishing(state.finishing)
        self._tau_slider.blockSignals(True)
        self._tau_slider.setValue(round(state.tau * _TAU_STEPS))
        self._tau_slider.blockSignals(False)
        self._refresh_tau_label()
        self._separation_slider.blockSignals(True)
        self._separation_slider.setValue(round(state.separation * _SEPARATION_STEPS))
        self._separation_slider.blockSignals(False)
        self._refresh_separation_label()
        self._show_editor_state()
        self._update_undo_buttons()
        self._edit_timer.start()

    def _show_editor_state(self) -> None:
        """Push the current table to the view, inspector and conflict banner."""
        table = self._history.current.table
        if self._zone_model is None or table is None:
            return
        self._corr_view.set_table(table)
        self._refresh_inspector()
        self._update_conflicts()

    def _refresh_inspector(self) -> None:
        model, table = self._zone_model, self._history.current.table
        zone_id = self._corr_view.selected_zone
        if model is None or table is None or not zone_id:
            self._inspector.set_zone(None, None, None, None)
            return
        self._inspector.set_zone(
            table.link(zone_id),
            model.source.zone(zone_id),
            model.reference,
            model.table.link(zone_id),
        )

    def _update_conflicts(self) -> None:
        if self._zone_model is None:
            self._show_conflicts([])
            return
        model = self._effective_model(self._zone_model)
        assert isinstance(model, ZoneTransferModel)
        self._show_conflicts(find_conflicts(model))

    def _show_conflicts(self, conflicts: list[Conflict]) -> None:
        self._corr_view.set_conflicts(conflicts)
        if not conflicts:
            self._conflict_banner.hide()
            return
        pairs = ", ".join(f"S{c.zone_a} et S{c.zone_b}" for c in conflicts)
        if len(conflicts) == 1:
            text = (
                f"⚠ Les zones {pairs} ont des couleurs trop proches pour être traitées "
                "différemment : la LUT va les mélanger."
            )
        else:
            text = (
                "⚠ Ces zones ont des couleurs trop proches pour être traitées différemment "
                f"(la LUT va les mélanger) : {pairs}."
            )
        text += " Donne-leur la même cible, ou verrouille l'une des deux."
        self._conflict_banner.setText(text)
        self._conflict_banner.show()

    def _setup_zone_editor(self, model: ZoneTransferModel) -> None:
        """Show the zones of a freshly fitted model, with its automatic table."""
        self._zone_model = model
        self._history.reset(replace(self._history.current, table=model.table))
        self._corr_view.set_data(model.source, model.reference, model.table, model.table)
        self._refresh_inspector()
        self._update_conflicts()
        self._update_undo_buttons()
        self._update_ai_buttons()
        pending, self._pending_recipe = self._pending_recipe, None
        if pending is not None:
            self.apply_recipe(pending, check_analysis=False)

    def _drop_zone_editor(self) -> None:
        """Forget the zones (images, method or zone count changed)."""
        self._zone_model = None
        self._corr_view.clear(_ANALYSING_MESSAGE if self._has_both_images() else _IDLE_MESSAGE)
        self._inspector.set_zone(None, None, None, None)
        self._show_conflicts([])
        self._on_zone_hovered("", 0)
        if self._history.current.table is not None:
            self._history.reset(replace(self._history.current, table=None))
        self._update_undo_buttons()
        self._update_ai_buttons()

    def _confirm_discard_edits(self, action: str) -> bool:
        """Ask before losing correspondences edited by hand."""
        table = self._history.current.table
        if table is None or not table.has_manual_links:
            return True
        answer = QMessageBox.question(
            self,
            "Correspondances modifiées",
            f"{action} relance l'analyse des zones : les correspondances que tu as "
            "modifiées seront perdues.\n\nContinuer ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _effective_model(self, model: TransferModel) -> TransferModel:
        """*model* with the user's correspondences and separation settings."""
        state = self._history.current
        if isinstance(model, LookTransferModel):
            return model.with_separation(state.separation)
        if not isinstance(model, ZoneTransferModel):
            return model
        table = state.table
        if table is None or not table.is_compatible(model.source.ids, model.reference.ids):
            table = model.table
        return model.with_table(table).with_params(ZoneBakeParams(tau=state.tau))

    def _tau(self) -> float:
        return self._tau_slider.value() / _TAU_STEPS

    def _refresh_tau_label(self) -> None:
        self._tau_value.setText(f"{self._tau():.2f}")

    def _separation(self) -> float:
        return self._separation_slider.value() / _SEPARATION_STEPS

    def _refresh_separation_label(self) -> None:
        self._separation_value.setText(f"{self._separation():.2f}")

    def _update_zone_panel_visibility(self) -> None:
        self._corr_group.setVisible(self._current_method == "zones")
        is_look = self._current_method == "look"
        for widget in (self._separation_label, self._separation_slider, self._separation_value):
            widget.setVisible(is_look)

    def _update_undo_buttons(self) -> None:
        self._undo_btn.setEnabled(self._history.can_undo)
        self._redo_btn.setEnabled(self._history.can_redo)

    def _method_key(self) -> str:
        """Model cache key: the method, plus the zone count for zones."""
        method = str(self._algo_combo.currentData())
        return f"zones:{self._zone_count}" if method == "zones" else method

    def _has_both_images(self) -> bool:
        return self._source_zone.has_image and self._reference_zone.has_image

    def _refresh_preview_if_ready(self) -> None:
        if self._has_both_images():
            self._run_transfer_preview()

    # ── Transfer preview ─────────────────────────────────────────────────

    def _run_transfer_preview(self) -> None:
        """Launch the TransferWorker for a live preview."""
        source = self._source_zone.image_rgb
        reference = self._reference_zone.image_rgb
        if source is None or reference is None:
            return

        method: str = str(self._algo_combo.currentData())
        lut_size: int = self._size_combo.currentData()

        # A newer request supersedes any preview still being computed
        self._retire_transfer_worker()

        self._status_bar.showMessage(f"Computing colour transfer preview ({method})…")
        self._progress_bar.setValue(0)

        key = self._method_key()
        raw_model = self._cached_model(source, reference, key)
        self._pending_model_key = (source, reference, key)
        model = None if raw_model is None else self._effective_model(raw_model)

        self._transfer_worker = TransferWorker(
            source,
            reference,
            method=method,
            lut_size=lut_size,
            params=self._adjustments.params(),
            finishing=self._history.current.finishing,
            model=model,
            zone_count=self._zone_count,
            prepare=partial(
                _prepare_model, self._history.current.tau, self._history.current.separation
            ),
        )
        self._transfer_worker.progress.connect(self._on_transfer_progress)
        self._transfer_worker.model_ready.connect(self._on_model_ready)
        self._transfer_worker.grid_ready.connect(self._on_grid_ready)
        self._transfer_worker.result_ready.connect(self._on_transfer_result)
        self._transfer_worker.error.connect(self._on_worker_error)
        self._transfer_worker.start()

    def _retire_transfer_worker(self) -> None:
        """Detach the current preview worker without blocking the GUI.

        Its signals are disconnected so a stale result can never overwrite
        a newer preview.  If it is still running, it stays referenced until
        it finishes: a ``QThread`` garbage-collected while running aborts
        the whole application ("QThread: Destroyed while thread is still
        running").
        """
        old = self._transfer_worker
        self._transfer_worker = None
        if old is None:
            return

        old.progress.disconnect(self._on_transfer_progress)
        old.model_ready.disconnect(self._on_model_ready)
        old.grid_ready.disconnect(self._on_grid_ready)
        old.result_ready.disconnect(self._on_transfer_result)
        old.error.disconnect(self._on_worker_error)

        if old.isRunning():
            self._stale_transfer_workers.append(old)
            old.finished.connect(partial(self._forget_stale_worker, old))

    def _forget_stale_worker(self, worker: TransferWorker) -> None:
        """Release a superseded preview worker once its thread has ended."""
        worker.wait()  # ``finished`` fires just before the thread fully exits
        if worker in self._stale_transfer_workers:
            self._stale_transfer_workers.remove(worker)

    def _cached_model(self, source: Any, reference: Any, method: str) -> TransferModel | None:
        """The fitted model for these exact images and method, if cached."""
        cache = self._model_cache
        if cache is None:
            return None
        c_source, c_reference, c_method, model = cache
        if c_source is source and c_reference is reference and c_method == method:
            return model
        return None

    def _on_model_ready(self, model: TransferModel) -> None:
        """Cache the model fitted by the current preview worker."""
        if self._pending_model_key is not None:
            self._model_cache = (*self._pending_model_key, model)
        if isinstance(model, ZoneTransferModel):
            self._setup_zone_editor(model)
        source = self._source_zone.image_rgb
        if isinstance(model, LookTransferModel) and source is not None:
            self._look_diagnostics = look_diagnostics(model, source)
            self._update_mode_hint()

    def _on_grid_ready(self, result: GridResult) -> None:
        self._last_grid_result = result

    def _on_transfer_progress(self, pct: int) -> None:
        self._progress_bar.setValue(pct)

    def _on_transfer_result(self, result: NDArray[np.uint8]) -> None:
        self._preview.set_image(result)
        self._progress_bar.setValue(100)
        grid_result = self._last_grid_result
        if grid_result is not None and grid_result.auto_smoothed:
            self._status_bar.showMessage(
                f"Preview ready — lissage relevé automatiquement à "
                f"{grid_result.smoothing_used:.2f} pour éviter des inversions de luminosité."
            )
        elif self._status_note:
            self._status_bar.showMessage(self._status_note)
        else:
            self._status_bar.showMessage("Preview ready — click 'Export .cube' to save.")
        self._status_note = ""
        logger.info("Transfer preview displayed")

    # ── Export ────────────────────────────────────────────────────────────

    def _run_export(self, output_path: Path) -> None:
        """Launch the ExportWorker to generate the .cube file."""
        source = self._source_zone.image_rgb
        reference = self._reference_zone.image_rgb
        if source is None or reference is None:
            return

        lut_size: int = self._size_combo.currentData()
        method: str = str(self._algo_combo.currentData())
        raw_model = self._cached_model(source, reference, self._method_key())

        # Disable export during processing
        self._export_btn.setEnabled(False)
        self._status_bar.showMessage(
            f"Exporting 3D LUT ({method}) to {output_path.name}…"
        )
        self._progress_bar.setValue(0)

        title = output_path.stem
        self._export_worker = ExportWorker(
            source_rgb=source,
            reference_rgb=reference,
            lut_size=lut_size,
            output_path=output_path,
            title=title,
            method=method,
            params=self._adjustments.params(),
            finishing=self._history.current.finishing,
            model=None if raw_model is None else self._effective_model(raw_model),
        )
        self._export_worker.progress.connect(self._on_export_progress)
        self._export_worker.result_ready.connect(self._on_export_result)
        self._export_worker.error.connect(self._on_worker_error)
        self._export_worker.finished.connect(self._on_export_finished)
        self._export_worker.start()

    def _on_export_progress(self, pct: int) -> None:
        self._progress_bar.setValue(pct)

    def _on_export_result(self, path_str: str) -> None:
        self._progress_bar.setValue(100)
        path = Path(path_str)
        self._status_bar.showMessage(f"Exported: {path.name}")
        logger.info("Exported to %s", path)

        msg = (
            f"3D LUT saved to:\n{path}\n\n"
            f"You can import this .cube file in Lightroom, "
            f"DaVinci Resolve, Premiere Pro, Photoshop, or any compatible application."
        )

        QMessageBox.information(
            self,
            "Export Complete",
            msg,
        )

    def _on_export_finished(self) -> None:
        """Re-enable the export button when the worker finishes."""
        self._update_export_button_state()

    # ── Error handling ───────────────────────────────────────────────────

    def _on_worker_error(self, message: str) -> None:
        """Display an error message from a worker."""
        self._progress_bar.setValue(0)
        self._status_bar.showMessage(f"Error: {message}")
        logger.error("Worker error: %s", message)

        QMessageBox.critical(self, "Error", message)
        self._update_export_button_state()

    # ── Helpers ──────────────────────────────────────────────────────────

    def _update_export_button_state(self) -> None:
        """Enable the export button only when both images are loaded."""
        both_loaded = self._source_zone.has_image and self._reference_zone.has_image
        not_exporting = (
            self._export_worker is None or not self._export_worker.isRunning()
        )
        is_enabled = both_loaded and not_exporting
        self._export_btn.setEnabled(is_enabled)

    def stop_workers(self, timeout_ms: int = 3000) -> None:
        """Wait for any running background workers to finish."""
        workers = [self._transfer_worker, self._export_worker, *self._stale_transfer_workers]
        for worker in workers:
            if worker is not None and worker.isRunning():
                worker.wait(timeout_ms)

    def closeEvent(self, event: Any) -> None:
        """Ensure background threads are safely terminated before closing."""
        self.stop_workers()
        super().closeEvent(event)


def _prepare_model(tau: float, separation: float, model: TransferModel) -> TransferModel:
    """Apply the user's separation settings to a freshly fitted model."""
    if isinstance(model, ZoneTransferModel):
        return model.with_params(ZoneBakeParams(tau=tau))
    if isinstance(model, LookTransferModel):
        return model.with_separation(separation)
    return model
