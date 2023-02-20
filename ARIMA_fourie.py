from all_functions import *
import warnings
from password import *

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
    product_list = tqdm(list(df_one_region.columns), desc='Формируется прогноз для', leave=True,
                        total=len(list(df_one_region.columns)))
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
                                                 today=last_date, region_code=region_code_dict, product_name=product,
                                                 region=region)
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

df_final.to_csv(f'Final/{today_date}.csv', index=False)
