# -*- coding: cp1251 -*-
import xml.etree.ElementTree as ET
from pandas import ExcelWriter, DataFrame
import requests, zipfile
import json
from typing import Dict, List
from io import BytesIO


class Date:
    def __init__(self, date: str):
        """
        :param date: дата
        """
        self.month, self.year = date.split(".")
        self.month = int(self.month)
        self.year = int(self.year)

        if self.month > 12 or self.month < 1:
            raise Exception

    def __sub__(self, other) -> int:
        """
        мы хотим по двум датам узнать смещение в месяцах
        :param other: объект класса Date
        :return: разница в месяцах
        """
        return (self.month - other.month + 1) + 12 * (self.year - other.year)

    def __add__(self, offset: int) -> str:
        """
        мы хотим по смещению и начальной дате узнать нужный месяц.год
        :param other: смещение
        :return: месяц.год
        """
        # (нач. месяц + смещение % 12) подхожит везде, кроме 12 месяца, поэтому ((нач. месяц + смещение -1) % 12)+1
        # нач. год увеличивается на 1, после того, как месяц становится 1, поэтому нач. год + ((смещение -1) // 12)
        return f"{(self.month + offset - 1) % 12 + 1}.{self.year + (self.month + offset - 1) // 12}"


TAG_NAMES = ["date", "DTPV", "district", "street", "house", "time", "kartId", "COORD_L", "COORD_W", "k_ul", "km", "m",
             "KTS", "KUCH", "POG", "RAN", "sdor", "spog", "osv", "factor", "ndu", "OBJ_DTP", "s_pch"]
# нужные нам теги
TRANSLATION = {'date': 'дата', 'DTPV': 'вид дтп', 'district': 'место', 'street': 'улица', 'house': 'дом',
               'time': 'время', 'kartId': 'номер', 'COORD_L': 'долгота', 'COORD_W': 'широта', 'k_ul': 'дорога',
               'km': 'километр', 'm': 'метр', 'KTS': 'количество транспортных средств', 'KUCH': 'количество участников',
               'POG': 'погибло', 'RAN': 'ранены', 'sdor': 'состояние дороги', 'spog': 'состояние погоды',
               'osv': 'освещение', 'factor': 'факторы',
               'ndu1': 'Недостатки транспортно-эксплуатационного содержания улично-дорожной сети1',
               'ndu2': 'Недостатки транспортно-эксплуатационного содержания улично-дорожной сети2',
               'ndu3': 'Недостатки транспортно-эксплуатационного содержания улично-дорожной сети3',
               'OBJ_DTP1': 'Объекты УДС вблизи места ДТП1', 'OBJ_DTP2': 'Объекты УДС вблизи места ДТП2',
               'OBJ_DTP3': 'Объекты УДС вблизи места ДТП3', 'OBJ_DTP4': 'Объекты УДС вблизи места ДТП4',
               'OBJ_DTP5': 'Объекты УДС вблизи места ДТП5', 'OBJ_DTP6': 'Объекты УДС вблизи места ДТП6',
               's_pch': 'состояние проезжей части'}
# перевод с тэга на русский, учитывая, что они несколько раз могут встречаться
MULTIPLE_VALUES = ['OBJ_DTP', 'ndu']  # те, кто несколько аз встречаются

REGIONS = {'Атнинский район': 92213, "Буинский район": 92218, "Сабинский район": 92252}
FIRST_REQUEST_URL = "http://stat.gibdd.ru/map/getDTPCardDataXML"
SECOND_REQUEST_URL = 'http://stat.gibdd.ru/getPDFbyId?data='


def save_to_excel(district: str, data: Dict[str, List[str]]) -> None:
    """
    сохраняет в эксель

    :param district: название эксель файла
    :param data: данные с карточек ДТП
    :return:
    """
    address = f'{district}\Список карточек ДТП.xml'
    tree = ET.parse(address)
    root = tree.getroot()
    read_xml(root, data)
    DataFrame(data).to_excel(f"{district}.xlsx", index=False)


def read_xml(xml, dtp_data):
    dict_for_multiple_values = dict(zip(MULTIPLE_VALUES, [0] * len(
        MULTIPLE_VALUES)))  # {'OBJ_DTP':0 , 'ndu':0} это для повторяющихся значений, например OBJ_DTP1,OBJ_DTP2
    for child in xml:  # проходим по всем детям
        if len(child):  # если у ребенка есть ребенок
            if child.tag == "tab":  # так еще и с тэгом таб
                for key, value in dtp_data.items():  # заполняем все ячейки " ", т.к. некоторые тэги могут не встретится
                    dtp_data[key].append("")
            read_xml(child, dtp_data)  # проходим по его детям
        else:  # у ребенка нет детей
            if child.tag in TAG_NAMES:  # если таг тот, который мы ищем
                if child.tag in MULTIPLE_VALUES:  # если он один из тех, кто встречается несколько раз в одном ребенке
                    dict_for_multiple_values[
                        child.tag] += 1  # увеличиваем на 1 цифру, которую мы ему припишем справа(т.к. их несколько)
                    count = dict_for_multiple_values[
                        child.tag]  # цифра детей, но еще бэз тэга (сделал так, чтобы код снизу получше читался)
                    dtp_data[TRANSLATION[f'{child.tag}{count}']][
                        -1] = child.text  # приписываем ему справа цифру, который он по счету
                else:
                    dtp_data[TRANSLATION[child.tag]][
                        -1] = child.text  # если дети не из MULTIPLE_VALUES, меняем пустой текст на текст детей


def get_files_id(region: str, start_date, end_date):  # добавил стартовую и конечную дату

    months = [f'"MONTHS:{start_date + offset}"' for offset in
              range(end_date - start_date)]  # пример: ["MONTHS:12.2021", "MONTHS:1.2022"]
    data = {"data": '{"date":[' + ', '.join(months) + '],' +
                    '"ParReg":"92",' +
                    '"order":{"type":"1","fieldName":"dat"},' +
                    f'"reg":"{REGIONS[region]}",' +
                    '"ind":"1","st":"1","en":"7"}'}

    reqnomer = requests.post(FIRST_REQUEST_URL, data=json.dumps(data), headers={'Content-type': 'application/json'})
    result_of_first = reqnomer.json()
    result_of_first["data"] = int(result_of_first["data"])
    reqxml = requests.get(SECOND_REQUEST_URL + str(result_of_first["data"]))
    file = zipfile.ZipFile(BytesIO(reqxml.content))
    file.extractall(district)


for district, district_id in REGIONS.items():
    start_date = Date("1.2021")  # начальная дата
    end_date = Date("12.2021")  # конечная дата
    print(f'Downloading {district}')
    get_files_id(district, start_date, end_date)
    data = {}
    print(f'{district} downloaded successfully')
    print(f'Saving {district} to excel')
    for key, value in TRANSLATION.items():
        data[value] = []
    save_to_excel(district, data)
    print(f'{district} saved successfully')
