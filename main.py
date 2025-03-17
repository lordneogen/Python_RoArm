import argparse
import math
import time

# from encodings.punycode import selective_len

import requests
import numpy as np
import onnxruntime as ort
from typing import Dict, Tuple, List, Optional


class RoboticArmController:
    """Класс для управления роботизированной рукой через ONNX модель"""

    JOINT_MAP: Dict[int, str] = {
        1: 'BASE_JOINT',  # Базовый сустав
        2: 'SHOULDER_JOINT',  # Плечевой сустав
        3: 'ELBOW_JOINT',  # Локтевой сустав
        4: 'EOAT_JOINT'  # Захват/Запястье
    }

    JOINT_LIMITS: Dict[int, Tuple[float, float]] = {
        1: (-90, 90),  # Диапазон для базового сустава
        2: (0, 90),  # Диапазон для плечевого сустава
        3: (0, 180),  # Диапазон для локтевого сустава
        4: (45, 135)  # Диапазон для захвата
    }

    MEMORY: List[List[float]]=[[0,0,0,0,4,4]]

    COORDINATES: List[int]=[0,0,0,0]

    def __init__(self, model_path: str, ip_address: Optional[str] = None, simulate: bool = False, use_speed:bool = False, speed : float=None, round_index:int = None, sleep_time:float=1,x_pos:int=0,z_pos:int=0):
        self.MEMORY = [[0.0, 0.0, 0.0, 0.0, 4.0, 4.0]]
        self.MEMORY[0][-1]=z_pos
        self.MEMORY[0][-2]=x_pos
        self.sleep_time = sleep_time
        self.round_index = round_index
        self.speed = speed
        self.use_speed = use_speed
        self.simulate = simulate
        self.ip = ip_address
        self.session = self._initialize_model(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        if not self.simulate and not self.ip:
            raise ValueError("IP-адрес обязателен в реальном режиме работы")

        if self.use_speed and not self.speed:
            raise ValueError("Введите параметр скорости")

        if not self.round_index:
            self.round_index = 2
    @staticmethod
    def _initialize_model(model_path: str) -> ort.InferenceSession:
        """Инициализация ONNX модели"""
        try:
            return ort.InferenceSession(model_path)
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели: {str(e)}")

    @staticmethod
    def _convert_range(
            value: float,
            old_range: Tuple[float, float],
            new_range: Tuple[float, float]
    ) -> float:
        """Преобразование значения из одного диапазона в другой"""
        old_min, old_max = old_range
        new_min, new_max = new_range
        return new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)

    def _send_command(self, joint: int, angle: float) -> None:
        """Отправка команды на сервер или вывод в консоль"""
        command = f'{{"T":108, "joint":{joint} ,"p":{angle},"i":0}}'

        if self.simulate:
            print(f"[Симуляция] Сустав {joint} ({self.JOINT_MAP[joint]}) → {angle}°")
            return

        url = f"http://{self.ip}/js?json={requests.utils.quote(command)}"
        try:
            response = requests.get(url, timeout=2)
            print(f"Сустав {joint} → {angle}°: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Ошибка соединения: {str(e)}")

    def _prepare_observation(self) -> np.ndarray:
        """Подготовка входных данных для модели"""
        observation = np.array(self.MEMORY[-1], dtype=np.float32)
        return observation.reshape(1, -1)



    def reset(self) -> None:
        command = f'{{"T":100 }}'

        if self.simulate:
            return

        url = f"http://{self.ip}/js?json={requests.utils.quote(command)}"
        try:
            response = requests.get(url, timeout=2)
        except requests.exceptions.RequestException as e:
            print(f"Ошибка соединения: {str(e)}")

    def run(self) -> None:
        """Основной цикл управления"""
        obs = self._prepare_observation()

        outputs = self.session.run(
            self.output_names,
            {self.input_name: obs}
        )
        _, _, continuous_actions, *_ = outputs

        new_rotation:List[float]=[0,0,0,0]

        for joint_idx in self.JOINT_MAP:
            normalized_value = continuous_actions[0][joint_idx - 1]
            if not self.use_speed:
                target_range = self.JOINT_LIMITS[joint_idx]
                angle = self._convert_range(
                    value=normalized_value,
                    old_range=(-1, 1),
                    new_range=target_range
                )
                new_rotation[joint_idx-1] = angle
                self.MEMORY.append(new_rotation)
                self._send_command(joint_idx, round(angle, self.round_index))
            else:
                target_range = self.JOINT_LIMITS[joint_idx]
                angle = self._convert_range(
                    value=normalized_value,
                    old_range=(-1, 1),
                    new_range=(-self.speed, self.speed)
                )
                angle=self.COORDINATES[joint_idx-1]+angle
                angle=min(angle,target_range[1])
                angle=max(angle,target_range[0])
                new_rotation[joint_idx-1] += angle
                self.MEMORY.append(new_rotation)
                self._send_command(joint_idx, round(angle, self.round_index))

    def run_loop(self) -> None:
        while True:
            obs = self._prepare_observation()

            outputs = self.session.run(
                self.output_names,
                {self.input_name: obs}
            )
            _, _, continuous_actions, *_ = outputs

            new_rotation: List[float] = [0, 0, 0, 0]

            for joint_idx in self.JOINT_MAP:
                normalized_value = continuous_actions[0][joint_idx - 1]
                if not self.use_speed:
                    target_range = self.JOINT_LIMITS[joint_idx]
                    angle = self._convert_range(
                        value=normalized_value,
                        old_range=(-1, 1),
                        new_range=target_range
                    )
                    new_rotation[joint_idx - 1] = angle
                    self.MEMORY.append(new_rotation)
                    self._send_command(joint_idx, round(angle, self.round_index))
                else:
                    target_range = self.JOINT_LIMITS[joint_idx]
                    angle = self._convert_range(
                        value=normalized_value,
                        old_range=(-1, 1),
                        new_range=(-self.speed, self.speed)
                    )
                    angle = self.COORDINATES[joint_idx - 1] + angle
                    angle = min(angle, target_range[1])
                    angle = max(angle, target_range[0])
                    new_rotation[joint_idx - 1] += angle
                    self.MEMORY.append(new_rotation)
                    self._send_command(joint_idx, round(angle, self.round_index))

            time.sleep(self.sleep_time)


from textual.app import App, ComposeResult
from textual.containers import Container, Grid
from textual.widgets import Button, Static, Header, Footer
from typing import Optional


class ArmControlApp(App):
    """Textual GUI для управления роботизированной рукой"""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2;
        grid-columns: 3fr 1fr;
    }

    Grid {
        border: round #666;
        padding: 1;
    }

    #info-panel {
        border: round #666;
        padding: 1;
    }

    Button {
        width: 100%;
        height: 100%;
    }
    """

    def __init__(self, controller, grid_size: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.controller = controller
        self.grid_size = grid_size

    def compose(self) -> ComposeResult:
        yield Header()
        yield Grid(*self.create_grid_buttons(), id="arm-grid")
        yield Container(
            Static("Текущие координаты:", id="coordinates"),
            Static("Последние углы суставов:", id="joints"),
            Static("Статус:", id="status"),
            Button("Стоп", id="stop-btn"),
            id="info-panel"
        )
        yield Footer()

    def create_grid_buttons(self):
        buttons = []
        for x in range(self.grid_size):
            for z in range(self.grid_size):
                btn = Button(f"X:{x} Z:{z}",
                             id=f"cell-{x}-{z}",
                             classes="cell-btn")
                buttons.append(btn)
        return buttons

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "stop-btn":
            self.exit()
        elif event.button.id.startswith("cell-"):
            _, x, z = event.button.id.split("-")
            self.update_coordinates(int(x), int(z))
            self.run_arm_control()

    def update_coordinates(self, x: int, z: int) -> None:
        self.controller.MEMORY[0][-2] = x
        self.controller.MEMORY[0][-1] = z
        self.query_one("#coordinates").update(f"Текущие координаты: X={x}, Z={z}")

    def update_joints_info(self) -> None:
        last_state = self.controller.MEMORY[-1]
        joints = "\n".join(
            f"{name}: {angle}°"
            for name, angle in zip(
                self.controller.JOINT_MAP.values(),
                last_state[:4]
            )
        )
        self.query_one("#joints").update(f"Последние углы:\n{joints}")

    def run_arm_control(self) -> None:
        try:
            self.controller.run()
            self.query_one("#status").update("Статус: Команда успешно отправлена")
            self.update_joints_info()
        except Exception as e:
            self.query_one("#status").update(f"Ошибка: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Управление роботизированной рукой через ONNX модель")
    parser.add_argument('--ip', type=str, default='192.168.4.1', help='IP-адрес устройства')
    parser.add_argument('--model', type=str, default='model.onnx', help='Путь к ONNX модели')
    parser.add_argument('--simulate', action='store_true', help='Режим симуляции без отправки команд на сервер')
    parser.add_argument('--use_speed', action='store_true', help='Использовать скорость для управления суставами')
    parser.add_argument('--speed', type=float, default=None, help='Скорость изменения угла суставов')
    parser.add_argument('--round_index', type=int, default=None, help='Количество знаков после запятой для округления углов')
    parser.add_argument('--sleep_time', type=float, default=1, help='Время задержки между командами в секундах')
    parser.add_argument('--x_pos', type=int, default=0, help='Начальная позиция по оси X')
    parser.add_argument('--z_pos', type=int, default=0, help='Начальная позиция по оси Z')
    parser.add_argument('--gui', action='store_true', help='Запустить графический интерфейс')
    parser.add_argument('--grid-size', type=int, default=4, help='Размер сетки для GUI')
    # args = parser.parse_args()
    args = parser.parse_args()

    try:
        controller = RoboticArmController(
            model_path=args.model,
            ip_address=None if args.simulate else args.ip,
            simulate=args.simulate,
            use_speed=args.use_speed,
            speed=args.speed,
            round_index=args.round_index,
            sleep_time=args.sleep_time,
            x_pos=args.x_pos,
            z_pos=args.z_pos
        )
        if args.gui:
            app = ArmControlApp(controller, grid_size=args.grid_size)
            app.run()
        else:
            controller.run()
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")


if __name__ == "__main__":
    main()