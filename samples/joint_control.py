# joint_control.py
import requests
import argparse

# Соответствие номеров суставов и параметров
JOINT_MAP = {
    1: 'x',  # Базовый сустав
    2: 'y',  # Плечевой сустав
    3: 'z',  # Локтевой сустав
    4: 't'  # Захват/Запястье
}


def send_command(ip, joint, angle):
    if joint not in JOINT_MAP:
        print(f"Ошибка: Недопустимый номер сустава {joint}. Допустимые значения: 1-4")
        return

    param = JOINT_MAP[joint]
    command = f'{{"T":1041, "{param}":{angle}}}'

    try:
        url = f"http://{ip}/js?json={requests.utils.quote(command)}"
        response = requests.get(url, timeout=2)
        print(f"Ответ от устройства: {response.text}")
    except Exception as e:
        print(f"Ошибка соединения: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Управление суставами RoArm-M2-S')
    parser.add_argument('ip', help='IP-адрес манипулятора')
    parser.add_argument('joint', type=int, help='Номер сустава (1-4)')
    parser.add_argument('angle', type=float, help='Угол/позиция для установки')

    args = parser.parse_args()

    send_command(args.ip, args.joint, args.angle)