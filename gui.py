import asyncio
import time
from datetime import datetime
from typing import List

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll, VerticalGroup, HorizontalGroup
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Footer, Header, Input, Static, Log
from main import RoboticArmController

class ButtonRow(HorizontalGroup):
    def __init__(self, i: int, max_i: int, *children: Widget):
        super().__init__(*children)
        self.i = i
        self.max_i = max_i

    def compose(self) -> ComposeResult:
        for x in range(self.max_i):
            yield Button(f"i={self.i},j={x}", id=f"idi{self.i}j{x}", variant="success")


class ButtonColumn(VerticalGroup):
    def __init__(self, max_i: int, *children: Widget):
        super().__init__(*children)
        self.max_i = max_i

    def compose(self) -> ComposeResult:
        for x in range(self.max_i):
            yield ButtonRow(x, max_i=self.max_i)


class ParametersPanel(VerticalScroll):
    def compose(self) -> ComposeResult:
        # Секция подключения
        yield Static("Подключение:", classes="section-title")
        yield Horizontal(
            Static("IP адрес:", classes="param-label"),
            Input(value="192.168.4.1", id="ip", classes="param-input"),
            classes="param-row"
        )
        yield Horizontal(
            Static("Модель:", classes="param-label"),
            Input(value="model.onnx", id="model", classes="param-input"),
            classes="param-row"
        )

        # Секция управления
        yield Static("Управление:", classes="section-title")
        yield Horizontal(
            Checkbox("Режим симуляции", id="simulate", classes="param-checkbox"),
            Static("(без реальных команд)", classes="param-hint"),
            classes="param-row"
        )
        yield Horizontal(
            Checkbox("Использовать скорость", id="use_speed", classes="param-checkbox"),
            classes="param-row"
        )
        yield Horizontal(
            Static("Скорость:", classes="param-label"),
            Input(placeholder="1.0",value="1.0", type="number", id="speed", classes="param-input"),
            classes="param-row"
        )

        # Секция параметров
        yield Static("Настройки:", classes="section-title")
        yield Horizontal(
            Static("Округление:", classes="param-label"),
            Input(placeholder="5",value="5", type="number", id="round_index", classes="param-input"),
            classes="param-row"
        )
        yield Horizontal(
            Static("Количество степов:", classes="param-label"),
            Input(placeholder="1000",value="1000", type="number", id="steps_count", classes="param-input"),
            classes="param-row"
        )
        yield Horizontal(
            Static("Задержка:", classes="param-label"),
            Input(value="1.0", type="number", id="sleep_time", classes="param-input"),
            Static("сек", classes="param-unit"),
            classes="param-row"
        )
        yield Horizontal(
            Static("Нормализация входа модели (углы):", classes="param-label"),
            Input(placeholder="1",value="1", type="number", id="input_norm", classes="param-input"),
            classes="param-row"
        )
        yield Horizontal(
            Checkbox("Бесконечный цикл", id="run_loop", classes="param-checkbox"),
            classes="param-row"
        )


class ManipulatorApp(App):
    """Приложение для управления манипулятором с логом сообщений"""

    BINDINGS = [("d", "toggle_dark", "Тёмная тема"),
                ("c", "clear_log", "Очистить лог")]

    CSS_PATH = "main.tcss"

    INDEX_X=0
    INDEX_Z=0
    MODEL:RoboticArmController
    _is_running = False  # Флаг выполнения операции

    def __init__(self,grid_size:int):
        super().__init__()
        self.grid_size = grid_size
        self.current_step = 0
        self.build=False

    async def run_model(self) -> None:
        try:
            if self.run_loop:
                await asyncio.to_thread(self.MODEL.run_loop)
            else:
                await asyncio.to_thread(self.MODEL.run)
        finally:
            self._is_running = False

    async def next_step(self) ->None:
        await asyncio.to_thread(self.MODEL.run)

    async def stop_model(self) -> None:
        await asyncio.to_thread(self.MODEL.stop_run)

    async def continue_model(self) -> None:
        await asyncio.to_thread(self.MODEL.continue_run)

    async def reset(self)->None:
        await asyncio.to_thread(self.MODEL.reset)

    async def get_log_step(self) -> None:
        self.query_one("#log",Log).clear()
        logs = await asyncio.to_thread(self.MODEL.get_memory_as_string)
        for log in logs:
            self.query_one("#log", Log).write_line(log)

    def combine_to_model(self) -> None:
        self.build=True
        self.MODEL=RoboticArmController(
            self.model_name,
            self.ip_address,
            self.simulation_mode,
            self.use_speed,
            self.speed,
            self.rounding,
            self.sleep_time,
            self.INDEX_X,
            self.INDEX_Z,
            self.grid_size,
            int(self.input_norm),
            self.max_step
        )
        self.MODEL.ManipulatorAPP=self

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Footer()
        yield Vertical(
            Horizontal(
                VerticalScroll(
                    Static("Позиция кубика:"),
                    ButtonColumn(5),
                    Button("Собрать модель", variant="primary", id="build_model"),
                    HorizontalGroup(
                        Button("Запустить модель", variant="primary", id="start_model"),
                        Button("Следующий ход модели", variant="success", id="next_step"),
                        Button("Обнулить", variant="success", id="reset"),
                        classes="buttons"
                    ),
                    HorizontalGroup(
                        Button("Остановить модель", variant="warning", id="stop_model"),
                        Button("Возобновить модель", variant="warning", id="run_model"),
                        Button("Логи модели", variant="warning", id="log_model"),
                        classes="buttons",
                    ),
                    classes="left-panel"
                ),
                ParametersPanel(classes="params-panel"),
            ),
            HorizontalGroup(
                Log(classes="debug-log",id="main"),
                Log(classes="debug-log",id="log"),
            ),
            classes="main-container"
        )

    # region [Input Properties]
    @property
    def ip_address(self) -> str:
        return self.query_one("#ip", Input).value

    @property
    def model_name(self) -> str:
        return self.query_one("#model", Input).value

    @property
    def simulation_mode(self) -> bool:
        return self.query_one("#simulate", Checkbox).value

    @property
    def use_speed(self) -> bool:
        return self.query_one("#use_speed", Checkbox).value

    @property
    def speed(self) -> float:
        return float(self.query_one("#speed", Input).value or 0)

    @property
    def rounding(self) -> int:
        return int(self.query_one("#round_index", Input).value or 2)

    @property
    def steps_count(self) -> int:
        return int(self.query_one("#steps_count", Input).value or 2)

    @property
    def sleep_time(self) -> float:
        return float(self.query_one("#sleep_time", Input).value or 1.0)

    @property
    def input_norm(self) -> float:
        return float(self.query_one("#input_norm", Input).value)

    @property
    def run_loop(self) -> bool:
        return self.query_one("#run_loop", Checkbox).value

    @property
    def max_step(self) -> int:
        return int(self.query_one("#steps_count", Input).value)
    # endregion

    def get_cube_button(self, i: int, j: int) -> Button:
        return self.query_one(f"#idi{i}j{j}", Button)

    def action_toggle_dark(self) -> None:
        self.theme = "textual-dark" if self.theme == "textual-light" else "textual-light"

    def action_clear_log(self) -> None:
        self.query_one("#main",Log).clear()

    def log_message(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.query_one("#main",Log).write_line(f"[{timestamp}] {message}")

    async def on_mount(self) -> None:
        self.set_cube_button_style(self.INDEX_X, self.INDEX_Z, "warning")

    def set_cube_button_style(self, i: int, j: int, variant: str) -> None:
        try:
            button = self.get_cube_button(i, j)
            button.variant = variant
            button.refresh()
        except Exception as e:
            self.log_message(f"Ошибка изменения стиля: {str(e)}")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if self._is_running:
            self.log_message("Операция уже выполняется, подождите")
            return

        button_id = event.button.id
        self._is_running = True
        try:
            if button_id == "start_model":
                self.log_message("[bold green]Запуск модели[/bold green]")
                await self.run_model()
            elif button_id == "stop_model":
                self.log_message("[bold yellow]Остановка модели[/bold yellow]")
                await self.stop_model()
            elif button_id == "next_step":
                self.log_message("[bold blue]Следующий шаг модели[/bold blue]")
                await self.next_step()
            elif button_id == "build_model":
                self.combine_to_model()
                self.log_message("[bold]Модель собрана![/bold]")
            elif button_id == "reset":
                await self.reset()
                self.log_message("[bold]Сброс состояния[/bold]")
            elif button_id == "log_model":
                await self.get_log_step()
            elif button_id == "run_model":
                await self.continue_model()
                self.log_message("[bold]Возобновление модели[/bold]")
            elif button_id and button_id.startswith("idi"):
                parts = button_id[3:].split("j")
                i, j = int(parts[0]), int(parts[1])
                self.set_cube_button_style(self.INDEX_X, self.INDEX_Z, "success")
                self.INDEX_X = i
                self.INDEX_Z = j
                self.set_cube_button_style(i, j, "warning")
                self.log_message(f"Выбрана позиция: [{i}, {j}]")
        except Exception as e:
            self.log_message(f"[red]Ошибка: {str(e)}[/red]")
        finally:
            self._is_running = False

    def on_input_changed(self, event: Input.Changed) -> None:
        self.log_message(f"Параметр {event.input.id} изменен на {event.input.value}")

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        state = "включен" if event.value else "выключен"
        self.log_message(f"Чекбокс {event.checkbox.id} {state}")

if __name__ == "__main__":
    app = ManipulatorApp(5)
    app.run()