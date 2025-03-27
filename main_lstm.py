import argparse
import math
import time
from datetime import datetime
import math
# from encodings.punycode import selective_len
# import serial
import requests
import numpy as np
import onnxruntime as ort
from typing import Dict, Tuple, List, Optional, Any

from numpy import ndarray, dtype, floating
from numpy._typing import _32Bit

from sympy.physics.quantum.tests.test_represent import x_bra

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
        4: (90, 180)  # Диапазон для захвата
    }

    MEMORY_AC:List[List[float]] = []

    MEMORY: List[List[float]] = [[0, 0, 0, 0, 4, 4]]  # Начальное состояние: углы, X, Z

    def _get_pose(self) -> ndarray[Any, dtype[Any]] | tuple[Any, Any, Any, Any] | None:
        command = f'{{"T":105 }}'
        if self.simulate:
            _last = self.MEMORY[-1].copy()
            _last[0] = self._convert_range(_last[0], self.JOINT_LIMITS[1],
                                           (-self.normalize_angle, self.normalize_angle))
            _last[1] = self._convert_range(_last[1], self.JOINT_LIMITS[2],
                                           (-self.normalize_angle, self.normalize_angle))
            _last[2] = self._convert_range(_last[2], self.JOINT_LIMITS[3],
                                           (-self.normalize_angle, self.normalize_angle))
            _last[3] = self._convert_range(_last[3], self.JOINT_LIMITS[4],
                                           (-self.normalize_angle, self.normalize_angle))

            observation = np.array(_last, dtype=np.float32)
            return observation.reshape(1, -1)

        url = f"http://{self.ip}/js?json={requests.utils.quote(command)}"

        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                data = response.json()

                b = data.get("b", 0)
                s = data.get("s", 0)
                e = data.get("e", 0)
                t = data.get("t", 0)

                b_d=math.degrees(b)
                s_d=math.degrees(s)
                e_d=math.degrees(e)
                t_d=math.degrees(t)



                _last = self.MEMORY[-1].copy()

                _last[0] = b_d
                _last[1] = s_d
                _last[2] = e_d
                _last[3] = t_d

                self.MEMORY[-1] = _last.copy()

                _last[0] = self._convert_range(b_d,self.JOINT_LIMITS[1],(-self.normalize_angle, self.normalize_angle))

                _last[1] = self._convert_range(s_d,self.JOINT_LIMITS[2],(-self.normalize_angle, self.normalize_angle))

                _last[2] = self._convert_range(e_d,self.JOINT_LIMITS[3],(-self.normalize_angle, self.normalize_angle))

                _last[3] = self._convert_range(t_d,self.JOINT_LIMITS[4],(-self.normalize_angle, self.normalize_angle))

                self.MEMORY_AC.append(_last.copy())

                # self.MEMORY[-1] = _last

                observation = np.array(_last, dtype=np.float32)

                return observation.reshape(1, -1)

                # print(f"Углы в радианах: b={b}, s={s}, e={e}, t={t}
            else:
                print(f"Ошибка: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Ошибка соединения: {str(e)}")




    def _normpos(self,x_pos:int,z_pos:int,max_pos:int) -> List[float]:

        res:List[float]=[0,0]
        res[0] = 2 * (float(x_pos) / float(max_pos - 1)) - 1
        res[1] = 2 * (float(z_pos) / float(max_pos - 1)) - 1
        return res

    def __init__(self, model_path: str, ip_address: Optional[str] = None, simulate: bool = False, use_speed: bool = False, speed: float = None, round_index: int = None, sleep_time: float = 1, x_pos: int = 0, z_pos: int = 0,grid_size:int = 5,normalize_angle:int = 1,max_step:int =100):
        self.step = 0
        self.lstm_memory = np.zeros((1, 1, 256), dtype=np.float32)

        self.max_step = max_step
        self.x_pos = self._normpos(x_pos,z_pos,grid_size)[1]
        self.z_pos = self._normpos(x_pos,z_pos,grid_size)[0]
        self.normalize_angle = normalize_angle
        self.MEMORY = [[0.0, 0.0, 90.0, 180.0, float(self.x_pos), float(self.z_pos)]]
        self.sleep_time = sleep_time
        self.round_index = round_index if round_index is not None else 2
        self.speed = speed
        self.use_speed = use_speed
        self.simulate = simulate
        self.ip = ip_address
        self.session = self._initialize_model(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        self.initial_memory=[[0.0, 0.0, 90.0, 180.0, float(self.x_pos), float(self.z_pos)]]
        self.on=True

        if not self.simulate and not self.ip:
            raise ValueError("IP-адрес обязателен в реальном режиме работы")
        if self.use_speed and not self.speed:
            raise ValueError("Введите параметр скорости")

    def get_memory_as_string(self) -> List[str]:
        output:List[str]=[]
        for step_num, step in enumerate(self.MEMORY):
            timestamp = datetime.now().strftime("%H:%M:%S")
            output.append(f"{timestamp} Шаг {step_num}:")
            for joint_idx in sorted(self.JOINT_MAP.keys()):
                joint_name = self.JOINT_MAP[joint_idx]
                angle = step[joint_idx - 1]
                rounded_angle = angle
                output.append(f"{timestamp} {joint_name}: {self.MEMORY_AC}")
                output.append(f"{timestamp} {joint_name}: {rounded_angle}°")
            x = step[4]
            z = step[5]
            # output.append()
            output.append(f"{timestamp} Позиция X: {x}, Z: {z}")
            output.append("-------------------")
        return output

    @staticmethod
    def _initialize_model(model_path: str) -> ort.InferenceSession:
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
        old_min, old_max = old_range
        new_min, new_max = new_range
        return new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)

    def _send_command(self, joint: int, angle: float) -> None:
        command = f'{{"T":121, "joint":{joint} ,"angle":{angle},"spd":10,"acc":10}}'

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

        return self._get_pose()

        # _last=self.MEMORY[-1].copy()
        # _last[0]=self._convert_range(_last[0],self.JOINT_LIMITS[1],(-self.normalize_angle,self.normalize_angle))
        # _last[1]=self._convert_range(_last[1],self.JOINT_LIMITS[2],(-self.normalize_angle,self.normalize_angle))
        # _last[2]=self._convert_range(_last[2],self.JOINT_LIMITS[3],(-self.normalize_angle,self.normalize_angle))
        # _last[3]=self._convert_range(_last[3],self.JOINT_LIMITS[4],(-self.normalize_angle,self.normalize_angle))
        #
        # observation = np.array(_last, dtype=np.float32)
        # return observation.reshape(1, -1)



    def reset(self) -> None:
        command = f'{{"T":100 }}'

        self.step=0
        self.lstm_memory =  np.zeros((1, 1, 256), dtype=np.float32)
        self.MEMORY=self.initial_memory.copy()

        if self.simulate:
            return

        url = f"http://{self.ip}/js?json={requests.utils.quote(command)}"
        try:
            response = requests.get(url, timeout=2)
        except requests.exceptions.RequestException as e:
            print(f"Ошибка соединения: {str(e)}")

    def continue_run(self) -> None:

        self.on = False

    def stop_run(self) -> None:

        self.on=False

    def run(self) -> None:
        obs = self._prepare_observation()

        inputs = {
            'obs_0': obs,  # Shape [1, 6]
            'recurrent_in': self.lstm_memory,  # Now shape [1, 1, 256]
        }

        outputs = self.session.run(None, inputs)

        continuous_actions = outputs[2]
        self.lstm_memory = outputs[5]

        self.MEMORY_AC.append(continuous_actions)

        new_rotation: List[float] = [0.0] * 4

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
                self._send_command(joint_idx, round(angle, self.round_index))
            else:
                target_range = self.JOINT_LIMITS[joint_idx]
                speed_adjusted = self._convert_range(
                    value=normalized_value,
                    old_range=(-1, 1),
                    new_range=(-self.speed, self.speed)
                )
                print(speed_adjusted)
                angle = self.MEMORY[-1][joint_idx - 1] + speed_adjusted
                angle = max(min(angle, target_range[1]), target_range[0])

                new_rotation[joint_idx - 1] = angle
                self._send_command(joint_idx, round(angle, self.round_index))


        new_rotation += [self.x_pos, self.z_pos]
        self.MEMORY.append(new_rotation)

    def run_loop(self) -> None:
        while True:

            if self.step>self.max_step or not self.on:
                break

            self.step+=1

            obs = self._prepare_observation()

            inputs = {
                'obs_0': obs,  # Shape [1, 6]
                'recurrent_in': self.lstm_memory,  # Now shape [1, 1, 256]
            }

            print(self.lstm_memory[0][0][0:5])

            outputs = self.session.run(None, inputs)

            continuous_actions = outputs[2]
            self.lstm_memory = outputs[5]

            new_rotation: List[float] = [0.0] * 4  # Инициализация под 4 сустава

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
                    self._send_command(joint_idx, round(angle, self.round_index))
                else:
                    target_range = self.JOINT_LIMITS[joint_idx]
                    speed_adjusted = self._convert_range(
                        value=normalized_value,
                        old_range=(-1, 1),
                        new_range=(-self.speed, self.speed)
                    )
                    angle = self.MEMORY[-1][joint_idx - 1] + speed_adjusted
                    angle = max(min(angle, target_range[1]), target_range[0])
                    new_rotation[joint_idx - 1] = angle
                    self._send_command(joint_idx, round(angle, self.round_index))

            # Добавляем [4, 4] после обработки всех суставов
            new_rotation += [self.x_pos, self.z_pos]
            self.MEMORY.append(new_rotation)

            time.sleep(self.sleep_time)

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
    parser.add_argument('--run_loop',action='store_true',help='Цикл')
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
        if args.run_loop:
            controller.run_loop()
        else:
            controller.run()
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")


if __name__ == "__main__":
    main()