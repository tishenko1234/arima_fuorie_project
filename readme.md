<h1 align="center">Приветствуем вас
<img src="https://github.com/blackcater/blackcater/raw/main/images/Hi.gif" height="32"/></h1>
<h1 align="center"><a href="https://github.com/AlchiProMent/GrForecast.git" target="_blank">"Прогнозирование цен"</a></h1>

<h4 align="center">2023 г</h4>


## Введение
Проект создан для автоматизированного расчета прогнозных значений временных рядов. Он позволяет в автоматическом режиме подключаться к базе данных формат PostgreSQL, выгружать данные из потока, предобрабатывать их и на основе полученного дата-сета формировать прогноз на указанного количество периодов
## Инструкция по установке
#### Для запуска кода необходимо установить Python 3.9 и выполнить следующие команды:
1. git clone http://... 
2. Создать виртуальное окружение командой 
* > py -m venv env для windows 
* > python -m venv venv для MacOs 
3. Активировать виртуальное окружение командой: 
* > venv\bin\Activate.ps1 для Windows 
* > venv\Scripts\activate для MacOs
4. Установить все необходимые библиотеки командой: 
* > py -m pip install -r requirements.txt
* > python3 -m pip3 install -r requirements.txt
5. Создать файл password.py, в котором будут указана необходимая информация для подключения к базе
* > def my_data():
    db_user = ''
    db_password = ''
    db_host = ''
    db_port = ''
    db_name = ''
    db_type = ''
    return db_name, db_user, db_host, db_password, db_port, db_type
6. Запустить программу командой: 
* > streamlit run [arima_fourie.py](arima_fourie.py)

## Структура репозитория 

1. [Файл all_functions.py](all_functions.py) содержит все разработанные функции
   * [get_df](all_functions.py) - функция, которая позволяет подключаться к БД. Для ее функционирования необходимо создать файл password.py, в котором будет функция с персональными данными для подключения к БД (см пункт 5 в инструкции по установке)
   * [weekly_data](all_functions.py) - функция, которая позволяет отфильтровать данные таким образом, чтобы на одну неделю приходилось только одно наблюдение. В случае когда в течение одной недели присутствует два и более наблюдения, вычисляется среднее значение. На выходе функция выдает словарь, ключом в котором является название региона, а содержимым является таблица формата pandas.DataFrame
   * [clean_data](all_functions.py) - данная функция убирает временной ряд в случае, если в нем, либо суммарно более 30% пропуском, либо присутствует более 4 пропусков подряд, либо отсутствуют данные на последнюю дату. На выходе выдается таблица формата pandas.DataFrame 
   * [arima_fourie](all_functions.py) - данная функция рассчитывает прогноз на 8 недель в перед и формирует таблицу, в которой следующие столбцы:
     * дата
     * название региона
     * код региона
     * название продукта
     * нижний 95% доверительный интервал
     * верхний 95% доверительный интервал
     * индекс прогноза - он отражает номер недели на которую рассчитывался прогноз. 0 - соответствует реальным данным на последний доступный период. 1 - соответствует прогнозному значению через неделю, 2 - через две недели и тд 
2. [Файл arima_fourie.py](arima_fourie.py) содержит основную программу, которая 
   * подключается к базе данных
   * загружает цены за 2020-2021 года из потока p_rind_week_price_history_2020_2021
   * загружает цены за 2022 год (и далее)
   * объединяет два массива цен в один
   * формирует прогноз на 8 недель вперед
   * добавляет прогноз в БД
3. [Файл arima_fourie_test_version.py](arima_fourie_test_version.py), который содержит тестовый вариант программы (прогноз строится лишь для двух регионов и двух продуктов)
4. [Файл requirements.txt](requirements.txt) содержит список библиотек, которые необходимо установить для функционирования основной программы

