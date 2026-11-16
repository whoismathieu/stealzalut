"""Themes (template + palette files), preferences and the appearance menu."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from stealzalut.config import Settings
from stealzalut.gui.main_window import MainWindow
from stealzalut.gui.preferences import IMAGE_BACKDROP, THEME, Preferences
from stealzalut.gui.theming import (
    DEFAULT_THEME,
    apply_theme,
    available_themes,
    build_stylesheet,
)
from stealzalut.gui.widgets.image_view import BACKDROPS, DEFAULT_BACKDROP, ImageView
from stealzalut.main import create_application
from stealzalut.utils.paths import get_styles_path


@pytest.fixture
def qapp() -> QApplication:
    return create_application([])


def _theme_file(folder: Path, key: str, tokens: dict[str, str], name: str = "") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {"tokens": tokens}
    if name:
        data["name"] = name
    (folder / f"{key}.json").write_text(json.dumps(data), encoding="utf-8")


class TestThemes:
    def test_shipped_themes_fill_every_token_of_the_template(self, tmp_path: Path) -> None:
        themes = available_themes([get_styles_path("themes")])
        assert next(iter(themes)) == DEFAULT_THEME
        assert len(themes) >= 3
        for theme in themes.values():
            sheet = build_stylesheet(theme, tmp_path / theme.key)
            assert "@" not in sheet
            assert theme.tokens["bg"] in sheet
            for icon in re.findall(r"url\((.+?)\)", sheet):
                assert Path(icon).is_file()

    def test_a_theme_only_lists_what_it_changes(self, tmp_path: Path) -> None:
        _theme_file(tmp_path, "mine", {"accent": "#ff0000"}, name="Le mien")
        themes = available_themes([get_styles_path("themes"), tmp_path])
        mine, default = themes["mine"], themes[DEFAULT_THEME]
        assert mine.name == "Le mien"
        assert mine.tokens["accent"] == "#ff0000"
        assert mine.tokens["bg"] == default.tokens["bg"]
        assert "#ff0000" in build_stylesheet(mine, tmp_path / "icons")

    def test_a_later_folder_replaces_a_theme_of_the_same_key(self, tmp_path: Path) -> None:
        _theme_file(tmp_path, "noir", {"bg": "#010203"})
        themes = available_themes([get_styles_path("themes"), tmp_path])
        assert themes["noir"].tokens["bg"] == "#010203"
        assert themes["noir"].name == "Noir"  # no name given: the file's

    @pytest.mark.parametrize(
        "content",
        [
            "not json",
            '{"name": "x"}',
            '{"tokens": {"bg": "red; } QWidget { color: blue"}}',
            '{"tokens": {"bg": 12}}',
        ],
    )
    def test_a_broken_theme_file_is_left_out(self, tmp_path: Path, content: str) -> None:
        tmp_path.joinpath("broken.json").write_text(content, encoding="utf-8")
        themes = available_themes([get_styles_path("themes"), tmp_path])
        assert "broken" not in themes and DEFAULT_THEME in themes

    def test_without_the_default_theme_nothing_can_be_built(self, tmp_path: Path) -> None:
        _theme_file(tmp_path, "mine", {"accent": "#ff0000"})
        with pytest.raises(ValueError, match="default theme"):
            available_themes([tmp_path])

    def test_apply_theme_styles_the_application(self, qapp: QApplication) -> None:
        folders = [get_styles_path("themes")]
        themes = available_themes(folders)
        applied = apply_theme(qapp, "noir", folders)
        assert applied is not None and applied.key == "noir"
        assert themes["noir"].tokens["bg"] in qapp.styleSheet()

    def test_an_unknown_theme_gives_the_default_one(self, qapp: QApplication) -> None:
        folders = [get_styles_path("themes")]
        applied = apply_theme(qapp, "dark", folders)
        assert applied is not None and applied.key == DEFAULT_THEME

    def test_a_failure_leaves_the_application_as_it_was(
        self, qapp: QApplication, tmp_path: Path
    ) -> None:
        qapp.setStyleSheet("")
        assert apply_theme(qapp, DEFAULT_THEME, [tmp_path]) is None
        assert qapp.styleSheet() == ""


class TestPreferences:
    def test_without_a_store_values_stay_in_memory(self) -> None:
        prefs = Preferences()
        assert prefs.get(THEME, "graphite") == "graphite"
        prefs.set(THEME, "noir")
        assert prefs.get(THEME, "graphite") == "noir"
        assert Preferences().get(THEME, "graphite") == "graphite"

    def test_a_store_keeps_them(self, tmp_path: Path) -> None:
        path = str(tmp_path / "prefs.ini")
        Preferences(QSettings(path, QSettings.Format.IniFormat)).set(IMAGE_BACKDROP, "white")
        again = Preferences(QSettings(path, QSettings.Format.IniFormat))
        assert again.get(IMAGE_BACKDROP, DEFAULT_BACKDROP) == "white"


class TestImageBackdrop:
    def test_the_six_greys_of_lightroom(self) -> None:
        assert [label for label, _ in BACKDROPS.values()] == [
            "Blanc",
            "Gris clair",
            "Gris moyen",
            "Gris foncé",
            "Gris très foncé",
            "Noir",
        ]
        assert [level for _, level in BACKDROPS.values()] == [255, 211, 129, 62, 34, 0]

    def test_a_view_takes_a_backdrop_and_refuses_an_unknown_one(self, qapp: QApplication) -> None:
        view = ImageView("vide")
        assert view.backdrop == DEFAULT_BACKDROP
        view.set_backdrop("white")
        assert view.backdrop == "white" and "rgb(255, 255, 255)" in view.styleSheet()
        view.set_backdrop("pink")
        assert view.backdrop == DEFAULT_BACKDROP

    def test_the_menu_ticks_the_current_grey_and_asks_for_another(self, qapp: QApplication) -> None:
        view = ImageView("vide")
        asked: list[str] = []
        view.backdrop_chosen.connect(asked.append)
        actions = {a.data(): a for a in view.context_menu().actions() if a.data()}
        assert list(actions) == list(BACKDROPS)
        assert [key for key, a in actions.items() if a.isChecked()] == [DEFAULT_BACKDROP]
        actions["medium"].trigger()
        assert asked == ["medium"]
        assert view.backdrop == DEFAULT_BACKDROP  # the listener decides


class TestAppearanceInTheWindow:
    @pytest.fixture
    def settings(self, tmp_path: Path) -> Settings:
        return Settings(lut_output_dir=tmp_path)

    def test_a_backdrop_picked_on_one_image_goes_to_the_three_and_is_kept(
        self, qapp: QApplication, settings: Settings
    ) -> None:
        prefs = Preferences()
        window = MainWindow(settings=settings, preferences=prefs)
        try:
            views = window._image_views()
            assert len(views) == 3
            actions = {a.data(): a for a in views[2].context_menu().actions() if a.data()}
            actions["light"].trigger()
            assert [view.backdrop for view in views] == ["light"] * 3
            assert prefs.get(IMAGE_BACKDROP, "") == "light"
        finally:
            window.stop_workers()
        again = MainWindow(settings=settings, preferences=prefs)
        try:
            assert again._source_zone.view.backdrop == "light"
        finally:
            again.stop_workers()

    def test_the_appearance_menu_lists_themes_and_backdrops(
        self, qapp: QApplication, settings: Settings
    ) -> None:
        prefs = Preferences()
        window = MainWindow(settings=settings, preferences=prefs)
        try:
            menu = window._appearance_menu()
            keys = [a.data() for a in menu.actions() if a.data()]
            assert keys[0] == DEFAULT_THEME and "noir" in keys
            assert keys[-len(BACKDROPS) :] == list(BACKDROPS)
            checked = [a.data() for a in menu.actions() if a.isChecked()]
            assert checked == [DEFAULT_THEME, DEFAULT_BACKDROP]

            next(a for a in menu.actions() if a.data() == "noir").trigger()
            assert prefs.get(THEME, "") == "noir"
            assert available_themes()["noir"].tokens["bg"] in qapp.styleSheet()
            checked = [a.data() for a in window._appearance_menu().actions() if a.isChecked()]
            assert checked[0] == "noir"
        finally:
            window.stop_workers()
            apply_theme(qapp, DEFAULT_THEME)

    def test_the_progress_bar_only_shows_while_something_runs(
        self, qapp: QApplication, settings: Settings
    ) -> None:
        window = MainWindow(settings=settings)
        try:
            assert window._progress_bar.isHidden()
            window._set_progress(40)
            assert not window._progress_bar.isHidden() and window._progress_bar.value() == 40
            window._set_progress(100)
            assert window._progress_bar.isHidden() and window._progress_bar.value() == 100
        finally:
            window.stop_workers()
