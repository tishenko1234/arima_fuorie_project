import pandas as pd
import missingno as mi
import numpy as np
import warnings
import psycopg2 as ps
from pmdarima import pipeline
from pmdarima import preprocessing as ppc
from pmdarima import arima
from datetime import date
from password import *
from tqdm import tqdm


def get_df(host, port, dbname, username, password, query):
    """Функция позволяет подключиться к базе данных и выгрузить данные через sql запрос"""
    try:
        with ps.connect(
                "host='{}' port={} dbname='{}' user={} password={}".format(host, port, dbname, username,
                                                                           password)) as conn:
            print("Соединение с базой данных установлено")
            try:
                print("Происходит выполнение sql-запроса...")
                dat = pd.read_sql_query(query, conn)
            except:
                print("Ошибка в sql запросе")
    except UnboundLocalError:
        print("Ошибка подключения к базе данных")
    return dat


def weekly_data(df_with_product_price):
    """ Функция позволяет преобразовать дата фрейм так, чтобы на одну неделю приходилось одно значение
        и формирует словарь, в котором содержится подобный дата-фрейм для каждого региона"""
    all_regions = {}
    # list(df_product_price['region_name'].unique())

    for region in ['Белгородская область']:
        all_products = {}
        all_products_test = {}
        product_list_weekly = tqdm(list(df_product_price['product'].unique()),
                                   desc='Преобразование даты в недельный формат',
                                   leave=True, total=len(list(df_product_price['product'].unique())))
        for product in product_list_weekly:
            product_list_weekly.set_description(
                f"""Преобразование даты в недельный формат в регионе {region}: {product}""")
            product_list_weekly.refresh()
            # выбор конкретного региона и продукта
            one_product_price = df_with_product_price.query(f"region_name == '{region}' and product == '{product}'")
            # преобразование данных в недельные
            one_product_price = one_product_price.groupby('date', as_index=False).agg({'avg_price': 'mean'})
            one_product_price = one_product_price.set_index('date')
            one_product_price = one_product_price.resample('W').mean()
            one_product_price = one_product_price.rename_axis('date').reset_index()
            one_product_price.sort_values(by='date', inplace=True)
            all_products[product] = one_product_price
            all_products_test[product] = one_product_price['avg_price'].values
        # Находим продукт с самым большим количеством данных
        biggest_length = 0
        biggest_product = ''
        for i in all_products_test.keys():
            if biggest_length < len(all_products_test[i]):
                biggest_length = len(all_products_test[i])
                biggest_product = i
        # формируем дата фрейм
        df = all_products[biggest_product]
        df = df.rename(columns={'avg_price': biggest_product})
        for prod in all_products.keys():
            df0 = all_products[prod]
            df0 = df0.rename(columns={'avg_price': prod})
            df = pd.merge(df, df0, how='left')
        all_regions[region] = df
    return all_regions


def clean_data(df_prices, nan_percent_total: int = 30, visualization=False):
    """ Функция позволяет удалить те продукты, по которым недостаточно данных и
     заполнить пропуски при помощи линейной интерполяции.
      Есть возможность визуализации до/после"""
    # Визуализация
    if visualization:
        mi.matrix(df_prices)
    # Проверка данных
    product_list = tqdm(df_prices.columns[1:], desc='Отчистка данных', leave=True)
    for prod in product_list:
        # Позволяет менять название индикатора продукта в зависимости от названия продукта
        product_list.set_description(f"Отчистка данных для продукта: {prod}")
        product_list.refresh()
        # Удаляем продукты по которым нет данных на последнюю дату
        if df_prices[prod].isna().iloc[-1]:
            df_prices = df_prices.drop(columns=prod)
        # Удаляем продукты, в которых пропусков больше какого-либо процента
        elif df_prices[prod].isna().mean() * 100 >= nan_percent_total:
            df_prices = df_prices.drop(columns=prod)
        # Удаляем продукты, где есть более 5% пропусков подряд
        else:
            nan_count = 0
            for price in df_prices[prod].isna():
                if price:
                    if nan_count > round(5 * len(df_prices[prod]) / 100):
                        df_prices.drop(columns=[prod], inplace=True)
                        break
                    else:
                        nan_count = nan_count + 1
                else:
                    if nan_count > round(5 * len(df_prices[prod]) / 100):
                        df_prices.drop(columns=[prod], inplace=True)
                        break
                    else:
                        nan_count = 0
    # Заполнение пропусков
    for prod in df_prices.columns[1:]:
        df_prices[prod].interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
    # Визуализация
    if visualization:
        mi.matrix(df_prices)
    return df_prices.set_index('date')


def arima_fourie(df_for_one_product, test_set=False, test_size=8,
                 data_depth=52, forecast_size=8, max_fourie_coeff=8,
                 today=date.today(), region_code={}, product_name="Название продукта"):
    """Функция позволяет рассчитать прогноз для временного ряда и
    вывести дата-фрейм с прогнозом и доверительными интервалами"""
    if test_set:
        train = df_for_one_product[:-test_size]
        test = df_for_one_product[-test_size:]
    else:
        train = df_for_one_product
    # Устанавливаем начальное значение критерия Акаике равным бесконечности
    aic_min = np.inf
    # Устанавливаем начальное значение коэффициента фурье равным одному
    best_fourie_coeff = 1
    # Формируем пустой словарь для сбора моделей
    all_models = {}
    for fourie_coeff in range(1, max_fourie_coeff + 1):
        # Настройка модели
        pipe = pipeline.Pipeline([
            ("fourier", ppc.FourierFeaturizer(m=data_depth, k=fourie_coeff)),
            ("arima", arima.AutoARIMA(stepwise=True, trace=False, error_action="ignore",
                                      seasonal=False,  # Потому что используем ряды Фурье
                                      suppress_warnings=True))])
        # Обучение модели
        pipe.fit(train)
        # Добавляем модель в словарь
        all_models[fourie_coeff] = pipe
        # Рассчитываем критерий Акаике
        aic_model = pipe.steps[-1][1].model_.aic()
        # Сравниваем качество моделей по критерию Акаике
        if aic_model < aic_min:
            aic_min = aic_model
            best_fourie_coeff = fourie_coeff
        else:
            break
    # Берем лучшую модель
    best_model = all_models[best_fourie_coeff]
    # Расчет прогнозов и доверительных интервалов
    predictions, conf_ints = best_model.predict(n_periods=forecast_size, fourier__n_periods=forecast_size,
                                                return_conf_int=True)
    region_code_number = region_code[region]
    date = pd.date_range(start=today + np.timedelta64(1, 'W'), periods=forecast_size, freq='W')
    df_one_product_prediction = pd.DataFrame({'date': date,
                                              'region_name': region,
                                              'region_code': region_code_number,
                                              'product': product_name,
                                              'prediction': predictions,
                                              'lower_conf_intl': conf_ints[:, 0],
                                              'upper_conf_intl': conf_ints[:, 1],
                                              'forecast period number': range(1, forecast_size + 1)})
    return df_one_product_prediction


warnings.filterwarnings('ignore')
# Получаем сегодняшнюю дату
today_date = date.today()
# Получаем данные подключения к базе данных
db_name, db_user, db_host, db_password, db_port, db_type = my_data()
# Формируем SQL запрос
sql = ("SELECT date,category product,value avg_price,region_name,region_code\n"
       "         FROM khd_kc.dal_data.p_rind_week_price_history_2020_2021\n"
       "         WHERE region_code IS NOT NULL\n"
       "         UNION SELECT date,product,avg_price,region_name,region_code\n"
       "         FROM khd_kc.dal_data.p_mprs_rst_002_a_food_price\n"
       "         WHERE region_code IS NOT NULL")
df_product_price = get_df(host=db_host, port=db_port, dbname=db_name, username=db_user, password=db_password, query=sql)

# Формируем словарь с кодом региона
region_code_dict = dict(zip(list(df_product_price.drop_duplicates(subset=['region_name'])['region_name']),
                            list(df_product_price.drop_duplicates(subset=['region_name'])['region_code'])))
# Преобразуем даты в формат даты
df_product_price['date'] = pd.to_datetime(df_product_price['date'])
# Преобразуем даты в формат даты
df_product_price_weekly = weekly_data(df_with_product_price=df_product_price)

df_final = pd.DataFrame()
for region in df_product_price_weekly.keys():
    print('\n', f'Обрабатывается {region}'.center(len(f'Обрабатывается {region}') + 60, '-'), end='\n')
    # Отчистим данные от не полных временных рядов
    df_one_region = clean_data(df_prices=df_product_price_weekly[region])
    product_list = tqdm(list(df_one_region.columns)[1:3], desc='Формируется прогноз для', leave=True,
                        total=len(list(df_one_region.columns)[1:3]))
    for product in product_list:
        # Позволяет менять название индикатора продукта в зависимости от названия продукта
        product_list.set_description(f"Формируется прогноз для: {product}")
        product_list.refresh()
        df_one_product = df_one_region[product]
        # Получаем информацию о последних данных, загруженных в бд
        last_period_data = pd.DataFrame(
            df_product_price.query(f'product == "{product}" and region_name == "{region}"')).sort_values(by='date',
                                                                                                         ascending=False).head(
            1)
        # Последняя дата добавления данных
        last_date = last_period_data['date'].values[0]
        # Цена последних добавленных данных
        last_price = last_period_data['avg_price'].values[0]
        # Рассчитываем прогноз
        df_one_product_prediction = arima_fourie(df_for_one_product=df_one_product, test_set=False, test_size=0,
                                                 data_depth=52, forecast_size=8, max_fourie_coeff=8,
                                                 today=last_date, region_code=region_code_dict, product_name=product)
        # Добавляем к прогнозу последние данные, загруженные в бд
        df_one_product_prediction = df_one_product_prediction.append(pd.DataFrame({'date': [last_date],
                                                                                   'region_name': [region],
                                                                                   'region_code': [
                                                                                       region_code_dict[region]],
                                                                                   'product': [product],
                                                                                   'prediction': [last_price],
                                                                                   'lower_conf_intl': [last_price],
                                                                                   'upper_conf_intl': [last_price],
                                                                                   'forecast period number': [0]}),
                                                                     ignore_index=True).sort_values(
            by='forecast period number', ascending=True).reset_index(drop=True)
        # Формируем финальный дата-фрейм
        df_final = pd.concat([df_final, df_one_product_prediction]).reset_index(drop=True)

df_final.to_excel(f'Final/{today_date}.xlsx', index=False)
