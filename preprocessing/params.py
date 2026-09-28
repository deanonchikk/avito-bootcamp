import re

TEXT_LABELS = [
    "Название услуги", "Специальность", "Куда выезжаете", "Ваши клиенты",
    "Кто оказывает услуги", "Где вы оказываете услуги", "Занятия",
    "Вид услуги", "Тип услуги", "Услуга", "Чем вы занимаетесь",
    "Марка авто", "Модель авто", "Тип автосервиса", "Тип обслуживаемой техники",
    "Где снимаете", "Дополнительно", "Как вы работаете",
]

FLAG_LABELS = [
    "Работа по договору", "Гарантия на выполнение работ", "Гарантия на работу", "Гарантия",
    "Готов закупить материалы", "Работаете с юрлицами и ИП", "Берёте ли срочные заказы",
    "Предоплата", "Работа в праздники и выходные", "Выезд",
]

NOISE_LABELS = [
    "Место оказания услуг", "Тип стоимости за услугу", "Тип стоимости за час", "Тип стоимости за м²",
    "Тип стоимости за единицу", "Начальная цена", "Опыт работы",
    "График работы, дни недели", "График работы от", "График работы до",
    "Время работы, с", "Время работы, до",
    "Время для связи, дни недели", "Время для связи от", "Время для связи до",
    "Время для связи, с", "Время для связи, до",
    "Стоимость", "Продолжительность", "Учебное учреждение", "Год окончания",
    "Бригада", "Выполняю заказы", "Рабочие дни",
    "Дни пн", "Дни вт", "Дни ср", "Дни чт", "Дни пт", "Дни сб", "Дни вс",
    "Срочная услуга (мультистатус)", "Рейтинг пользователя",
    "Минимальная сумма заказа", "Минимальное время аренды", "Минимальное время заказа", "Минимальное время",
    "Признак предзаполнения прайс листа", "Признак мигрированного прайс листа", "Признак",
    "Можно со своими запчастями", "Камера наблюдения в ремонтной зоне", "Камера наблюдения",
]

_ALL_LABELS = sorted(set(TEXT_LABELS + FLAG_LABELS + NOISE_LABELS), key=lambda label: (-len(label), label))
_PATTERN = re.compile("(" + "|".join(re.escape(l) for l in _ALL_LABELS) + ")")


def extract_semantic_params(text: object) -> str:
    if not isinstance(text, str) or not text:
        return ""
    parts = _PATTERN.split(text)
    out = []
    seen_flags = set()
    for i in range(1, len(parts), 2):
        label = parts[i]
        value = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if label in TEXT_LABELS:
            if value and value != "Своя услуга":
                out.append(value)
        elif label in FLAG_LABELS and label not in seen_flags:
            out.append(label)
            seen_flags.add(label)
    result = []
    for v in out:
        if not result or result[-1] != v:
            result.append(v)
    return " ".join(result)


def extract_label_value(text: object, label: str) -> str | None:
    if not isinstance(text, str) or not text:
        return None
    parts = _PATTERN.split(text)
    for i in range(1, len(parts), 2):
        if parts[i] == label:
            return parts[i + 1].strip() if i + 1 < len(parts) else None
    return None
